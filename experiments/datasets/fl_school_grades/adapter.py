"""Florida school grades (FLDOE): A-F bands on the percent of points, cutoffs by school type.

Each Florida public school gets a letter grade from the percent of possible points it
earns: up to 12 components (achievement, learning gains, acceleration, graduation),
each worth up to 100 points, summed and divided by 100 x the number of components with
enough data. The percent is rounded half up to a whole number and compared with the
year's grading scale for the school's type (elementary, middle, high, combination).

Q = the unrounded percent of points, x = total points / number of components, centred at
    the unit's own cutoff for the chosen grade boundary. Because the published percent is
    round-half-up(x), "published percent >= c" is exactly "x >= c - 0.5", so
    Q = x - (c - 0.5) and D = 1{Q >= 0} = the school gets the higher of the two grades.
    For the C/D and D/F boundaries the policy treatment (state intervention) is 1 - D.
Cutoffs: CUTOFFS[year][type] = minimum published percent for A, B, C, D, transcribed from
    FLDOE's yearly "Guide to Calculating School Grades" and, for 2022-23 onward, checked
    against the scale printed in each spreadsheet (``rule_check()``).
X = school characteristics measured before the grade or independently of it: percent
    economically disadvantaged, percent minority, log October membership, charter,
    Title I, alternative/ESE center, collocated, virtual, number of graded components,
    school type, year and district indicators.
Y = load(): next grading year's unrounded percent of points, matched by district, school
    and virtual-provider number. Other loaders: next ELA / math achievement, next grade
    (A=4 ... F=0), and not being graded next year (closure, merger or renumbering).

Sample: school-years graded A-F in 2016-17, 2017-18, 2018-19, 2021-22, 2022-23, 2023-24
and 2024-25 (spring years 2017-2019 and 2022-2025). Excluded: "I" (incomplete, no
percent), 2020-21 (grades only for schools that opted in, a choice made knowing the
score), 2014-15 and 2015-16 (no membership file on the CDN; 2014-15 also lacks learning
gains), 2025-26 (no next-year outcome yet). 2019-20 has no grades (COVID-19), so the
outcome for 2018-19 comes from 2020-21, and the outcome for 2021-22 from 2022-23, the
informational-baseline year on the new FAST tests (achievement components only).
``extras["y_year"]`` records the outcome year.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample
from experiments.datasets.fl_school_grades._spreadsheet import read, sheet_names
from experiments.datasets.fl_school_grades.download import MEMBERSHIP, RAW, SPREADSHEETS

HERE = Path(__file__).parent
PROCESSED = HERE / "data" / "processed"
CACHE = PROCESSED / "fl_school_grades_panel.csv.gz"

TYPES = {1: "elementary", 2: "middle", 3: "high", 4: "combination"}
_LEGACY = (62, 54, 41, 32)          # 2014-15 to 2021-22, every type; elementary since
_MHC = (64, 57, 44, 34)             # middle / high / combination, 2023-24 and 2024-25
# Minimum published percent of points for A, B, C, D (F below D), by spring year and type.
# Sources: SchoolGradesCalcGuide{15..26}.pdf, section "School Grading Percentages".
CUTOFFS = {
    **{y: dict.fromkeys(TYPES, _LEGACY) for y in (2015, 2016, 2017, 2018, 2019, 2021, 2022)},
    2023: {1: (62, 50, 33, 22), 2: (68, 58, 40, 31), 3: (70, 60, 40, 23), 4: (67, 56, 35, 23)},
    2024: {1: _LEGACY, 2: _MHC, 3: _MHC, 4: _MHC},
    2025: {1: _LEGACY, 2: _MHC, 3: _MHC, 4: _MHC},
    2026: {1: _LEGACY, 2: _MHC, 3: (65, 60, 45, 35), 4: (65, 60, 45, 35)},
}
BOUNDARIES = {"AB": 0, "BC": 1, "CD": 2, "DF": 3}
MAIN_BOUNDARY = "BC"   # the most school-years within 2 points of the cutoff (README)
MAIN_YEARS = (2017, 2018, 2019, 2022, 2023, 2024, 2025)
GRADE_POINTS = {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0, "F": 0.0}

_COLS = {
    "District Number": "district", "School Number": "school",
    "Virtual Provider Number": "vpn", "School Name": "name",
    "Grade 3 English Language Arts Achievement": "ela3_ach",
    "English Language Arts Achievement": "ela_ach",
    "English Language Arts Learning Gains": "ela_lg",
    "English Language Arts Learning Gains of the Lowest 25%": "ela_lg25",
    "Mathematics Achievement": "math_ach", "Mathematics Learning Gains": "math_lg",
    "Mathematics Learning Gains of the Lowest 25%": "math_lg25",
    "Science Achievement": "sci_ach", "Social Studies Achievement": "ss_ach",
    "Middle School Acceleration": "ms_accel", "Total Points Earned": "points",
    "Total Components": "n_components", "Percent of Total Possible Points": "pct_points",
    "Percent Tested": "pct_tested", "Was the collocated rule used?": "collocated",
    "Charter School": "charter", "Title I": "title1", "Title 1": "title1",
    "Alternative/ESE Center School": "alt_ese", "School Type": "school_type",
    "Percent of Minority Students": "pct_minority",
    "Percent of Economically Disadvantaged Students": "pct_econ_dis",
    "Did the school submit an opt-in request?": "opt_in",
}
_PREFIX = {"Graduation Rate": "grad_rate", "College and Career Acceleration": "cc_accel"}
COMPONENTS = ["ela3_ach", "ela_ach", "ela_lg", "ela_lg25", "math_ach", "math_lg",
              "math_lg25", "sci_ach", "ss_ach", "ms_accel", "grad_rate", "cc_accel"]
ACHIEVEMENT = ["ela3_ach", "ela_ach", "math_ach", "sci_ach", "ss_ach"]
GAINS = ["ela_lg", "ela_lg25", "math_lg", "math_lg25"]


# ------------------------------------------------------------------------ raw files

def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.strip().replace({"": None, "None": None}),
                         errors="coerce")


def _flag(s: pd.Series) -> pd.Series:
    v = s.fillna("").astype(str).str.strip().str.upper()
    return v.isin(["Y", "YES"]).astype(float)


def _sheet_frame(rows: list) -> tuple[pd.DataFrame, list]:
    h = next(k for k, r in enumerate(rows) if r and r[0] == "District Number")
    hdr = [" ".join(str(x).split()) if x is not None else f"col{j}"
           for j, x in enumerate(rows[h])]
    body = [r + [None] * (len(hdr) - len(r)) for r in rows[h + 1:]]
    df = pd.DataFrame(body, columns=hdr)
    df = df[df["District Number"].notna() & (df["District Number"].astype(str).str.strip() != "")]
    notes = [x for r in rows[:h] for x in r if isinstance(x, str)]
    return df, notes


def _read_year(year: int) -> pd.DataFrame:
    path = RAW / SPREADSHEETS[year]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run python -m experiments.datasets.fl_school_grades.download")
    out = []
    for i in range(len(sheet_names(path))):
        df, _ = _sheet_frame(read(path, i))
        cols = {}
        for c in df.columns:
            if c in _COLS:
                cols[c] = _COLS[c]
            elif c in (f"Grade {year}", f"Informational Baseline Grade {year}"):
                cols[c] = "grade"
            else:
                cols.update({c: v for k, v in _PREFIX.items() if c.startswith(k)})
        out.append(df[list(cols)].rename(columns=cols))
    df = pd.concat(out, ignore_index=True)
    res = pd.DataFrame({"year": year, "district": _num(df.district).astype(int),
                        "school": _num(df.school).astype(int),
                        "vpn": _num(df["vpn"]).fillna(0).astype(int) if "vpn" in df else 0,
                        "name": df["name"].astype(str).str.strip(),
                        "school_type": _num(df.school_type).astype(int),
                        "grade": df.grade.fillna("").astype(str).str.strip()})
    for c in COMPONENTS + ["points", "n_components", "pct_points", "pct_tested",
                           "pct_minority", "pct_econ_dis"]:
        res[c] = _num(df[c]) if c in df else np.nan
    for c in ["collocated", "charter", "title1"]:
        res[c] = _flag(df[c])
    # Coded N / A (alternative school) / E (ESE center school).
    res["alt_ese"] = df.alt_ese.fillna("").astype(str).str.strip().str.upper().isin(
        ["A", "E", "Y", "YES"]).astype(float)
    res["opt_in"] = _flag(df["opt_in"]) if "opt_in" in df else np.nan
    return res


def _membership(year: int) -> pd.DataFrame:
    """October (Survey 2) membership by school for the school year ending in `year`."""
    path = RAW / "membership" / MEMBERSHIP[year]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run python -m experiments.datasets.fl_school_grades.download")
    for i in range(len(sheet_names(path))):
        rows = read(path, i)
        for h, r in enumerate(rows[:10]):
            cells = [str(x).strip().upper() for x in r if x is not None]
            if "SCHOOL #" in cells and len(cells) == 5:
                df = pd.DataFrame([x[:5] for x in rows[h + 1:] if x and len(x) >= 5],
                                  columns=["district", "dname", "school", "sname", "n"])
                df = pd.DataFrame({"district": _num(df.district), "school": _num(df.school),
                                   "enrollment": _num(df.n),
                                   "masked": df.n.astype(str).str.contains(r"\*")})
                df = df.dropna(subset=["district", "school"])
                df = df[(df.district > 0) & ~df.duplicated(["district", "school"], keep=False)]
                df["year"] = year
                return df.astype({"district": int, "school": int})
    raise ValueError(f"no school-total sheet in {path}")


def build() -> pd.DataFrame:
    """Every school-year in the 2014-15 to 2025-26 spreadsheets (2013-14 is the old
    model and is skipped), with the unrounded percent, the computed grade, membership
    and the next grading year's outcomes."""
    if not CACHE.exists():
        df = _build()
        PROCESSED.mkdir(parents=True, exist_ok=True)
        df.to_csv(CACHE, index=False)
    p = pd.read_csv(CACHE, keep_default_na=False, na_values=[""])
    for c in ["name", "grade", "grade_computed", "y_grade"]:
        p[c] = p[c].fillna("").astype(str)
    p["masked"] = p.masked.astype(str).eq("True")
    return p


def _build() -> pd.DataFrame:
    years = sorted(y for y in SPREADSHEETS if y >= 2015)
    p = pd.concat([_read_year(y) for y in years], ignore_index=True)
    p["x"] = p.points / p.n_components
    p["pct_rounded"] = np.floor(p.x + 0.5)
    cut = np.array([CUTOFFS[y][t] for y, t in zip(p.year, p.school_type)], dtype=float)
    for j, g in enumerate("ABCD"):
        p[f"cut_{g}"] = cut[:, j]
    pr = p.pct_points.to_numpy()
    p["grade_computed"] = np.select([pr >= cut[:, k] for k in range(4)], list("ABCD"), "F")
    p.loc[p.pct_points.isna(), "grade_computed"] = ""
    # Percent minority is missing from the 2024-25 and 2025-26 files: carry the school's
    # latest earlier value forward and flag it.
    p = p.sort_values(["district", "school", "vpn", "year"])
    p["minority_carried"] = (p.pct_minority.isna() & (p.year >= 2025)).astype(float)
    p["pct_minority"] = np.where(p.year >= 2025,
                                 p.groupby(["district", "school", "vpn"]).pct_minority.ffill(),
                                 p.pct_minority)
    mem = pd.concat([_membership(y) for y in sorted(MEMBERSHIP)], ignore_index=True)
    p = p.merge(mem[["year", "district", "school", "enrollment", "masked"]],
                on=["year", "district", "school"], how="left")
    # Next grading year: 2018-19 -> 2020-21 (no grades in 2019-20).
    nxt = dict(zip(years[:-1], years[1:]))
    p["y_year"] = p.year.map(nxt)
    key = ["district", "school", "vpn"]
    n = p[key + ["year", "x", "pct_points", "grade", "ela_ach", "math_ach"]].rename(
        columns={"year": "y_year", "x": "y_pct_points", "pct_points": "y_pct_published",
                 "grade": "y_grade", "ela_ach": "y_ela_ach", "math_ach": "y_math_ach"})
    p = p.merge(n, on=key + ["y_year"], how="left", indicator=True)
    p["y_closed"] = np.where(p.y_year.notna(), (p._merge == "left_only").astype(float), np.nan)
    p["y_grade_points"] = p.y_grade.map(GRADE_POINTS)
    p = p.drop(columns="_merge")
    lag = p[key + ["year", "x"]].rename(columns={"x": "q_lag"})
    lag["year"] = lag.year.map(nxt)
    return p.merge(lag.dropna(subset=["year"]).astype({"year": int}), on=key + ["year"], how="left")


# --------------------------------------------------------------------------- checks

def _note_cutoffs(year: int) -> dict:
    """Grading scales printed in the 2022-23 to 2025-26 spreadsheets, by school type."""
    words = {"elementary": 1, "middle": 2, "high": 3, "combination": 4}
    pat = re.compile(r"grading scale for ([a-z, ]+?) schools is as follows:\s*A=(\d+)% of "
                     r"points or greater; B=(\d+)% to \d+% of points; C=(\d+)% to \d+% of "
                     r"points; D=(\d+)%")
    path = RAW / SPREADSHEETS[year]
    found = {}
    for i in range(len(sheet_names(path))):
        _, notes = _sheet_frame(read(path, i))
        for m in pat.finditer(" ".join(" ".join(notes).split())):
            for w in re.split(r",\s*|\s+and\s+|,?\s*and\s+", m.group(1)):
                if w.strip() in words:
                    found[words[w.strip()]] = tuple(int(m.group(k)) for k in range(2, 6))
    return found


def rule_check() -> pd.DataFrame:
    """Published grade against the grade computed from the percent and CUTOFFS, by year,
    for every school with a published A-F grade. Also checks that the published percent
    is round-half-up(points / components) and, for 2023 onward, that CUTOFFS matches the
    scale printed in the spreadsheet."""
    p = build()
    rows = []
    for y, g in p.groupby("year"):
        m = g.grade.isin(list("ABCDF"))
        notes = _note_cutoffs(y) if y >= 2023 else {}
        rows.append({
            "year": y, "rows": len(g), "graded": int(m.sum()),
            "incomplete": int((g.grade == "I").sum()),
            "no_grade": int((g.grade == "").sum()),     # 2020-21: did not opt in
            "agree": float((g.grade[m] == g.grade_computed[m]).mean()),
            "pct_is_rounded_x": float((g.pct_rounded == g.pct_points)[g.pct_points.notna()].mean()),
            "scale_in_file": (all(notes.get(t) == CUTOFFS[y][t] for t in TYPES) if notes
                              else None),
        })
    return pd.DataFrame(rows)


# -------------------------------------------------------------------------- samples

def _design(df: pd.DataFrame) -> tuple[np.ndarray, list]:
    X = pd.DataFrame(index=df.index)
    X["pct_econ_dis"] = df.pct_econ_dis.fillna(df.pct_econ_dis.median())
    X["econ_missing"] = df.pct_econ_dis.isna().astype(float)
    X["pct_minority"] = df.pct_minority.fillna(df.pct_minority.median())
    X["minority_missing"] = df.pct_minority.isna().astype(float)
    # 2024-25 minority shares are the school's 2023-24 values (minority_carried == year_2025).
    # Membership below 10 is masked with "*" in the file; it is set to 5.
    n = df.enrollment.where(~df.masked, 5.0).where(lambda v: v > 0)
    X["log_enrollment"] = np.log(n).fillna(np.log(n).median())
    X["enrollment_missing"] = n.isna().astype(float)
    for c in ["charter", "title1", "alt_ese", "collocated"]:
        X[c] = df[c]
    X["virtual"] = ((df.vpn > 0) | df.name.str.contains("VIRTUAL", na=False)).astype(float)
    X["n_components"] = df.n_components
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
    df = p[p.year.isin(years) & p.grade.isin(list("ABCDF")) & p.x.notna()
           & p[y_col].notna()]
    return df.reset_index(drop=True)


def _sample(boundary: str = MAIN_BOUNDARY, y_col: str = "y_pct_points") -> RDDSample:
    j = BOUNDARIES[boundary]
    df = _frame(y_col)
    hi, lo = "ABCD"[j], "BCDF"[j]
    c = df[f"cut_{hi}"].to_numpy(float) - 0.5
    X, names = _design(df)
    # A group is a (year, school type) cell; its cutoff is on the unrounded scale.
    group = (df.year.astype(str) + "-" + df.school_type.map(TYPES)).to_numpy()
    group_cutoffs = {f"{y}-{TYPES[t]}": CUTOFFS[y][t][j] - 0.5
                     for y in sorted(df.year.unique()) for t in TYPES}
    return RDDSample(
        Q=df.x.to_numpy(float) - c, X=X, Y=df[y_col].to_numpy(float), threshold=0.0,
        name=f"fl_school_grades_{boundary}_{y_col}", feature_names=names,
        description=(f"Florida school grades, spring years {', '.join(map(str, MAIN_YEARS))}. "
                     f"Q = unrounded percent of points minus (own {hi}-cutoff - 0.5); "
                     f"treatment = 1{{Q >= 0}} (grade {hi} or better rather than {lo}); "
                     f"Y = {y_col} (next grading year)."),
        citation="Florida Department of Education, School Grades 2016-17 to 2025-26 and "
                 "Membership in Florida Public Schools (Survey 2)",
        treatment_rule=lambda Q: (Q >= 0).astype(int),
        extras={
            "group": group, "group_cutoffs": group_cutoffs,
            "boundary": boundary, "cutoff_table": CUTOFFS,
            "year": df.year.to_numpy(int), "school_type": df.school_type.to_numpy(int),
            "district": df.district.to_numpy(int), "school": df.school.to_numpy(int),
            "q_raw": df.x.to_numpy(float), "pct_published": df.pct_points.to_numpy(float),
            "grade_published": df.grade.to_numpy(str),
            "cutoffs": df[["cut_A", "cut_B", "cut_C", "cut_D"]].to_numpy(float) - 0.5,
            "eligible_published": df.grade.isin(list("ABCD"[:j + 1])).to_numpy(float),
            "y_year": df.y_year.to_numpy(int), "q_lag": df.q_lag.to_numpy(float),
            "pct_tested": df.pct_tested.to_numpy(float),
            "informational": (df.year == 2023).to_numpy(float),
        },
    )


def load() -> RDDSample:
    """B/C boundary; Y = next grading year's unrounded percent of points."""
    return _sample(MAIN_BOUNDARY)


def load_ab() -> RDDSample:
    return _sample("AB")


def load_bc() -> RDDSample:
    return _sample("BC")


def load_cd() -> RDDSample:
    return _sample("CD")


def load_df() -> RDDSample:
    return _sample("DF")


def load_next_ela() -> RDDSample:
    return _sample(MAIN_BOUNDARY, "y_ela_ach")


def load_next_math() -> RDDSample:
    return _sample(MAIN_BOUNDARY, "y_math_ach")


def load_next_grade() -> RDDSample:
    """Y = next grade (A=4 ... F=0). Missing for 2018-19, since 2020-21 grades were
    optional, so that year mostly drops out."""
    return _sample(MAIN_BOUNDARY, "y_grade_points")


def load_closed() -> RDDSample:
    """Y = 1 if the school has no row in the next grading year's file."""
    return _sample(MAIN_BOUNDARY, "y_closed")


def load_two_score(boundary: str = MAIN_BOUNDARY) -> RDDSample:
    """Achievement against learning gains, for elementary schools graded on exactly the
    seven components ELA / math / science achievement and the four learning-gains
    components (2016-17 to 2018-19 and 2021-22). Their percent is (3 * achievement mean
    + 4 * gains mean) / 7, a fixed-weight sum with the elementary cutoffs 62/54/41/32."""
    j = BOUNDARIES[boundary]
    df = _frame("y_pct_points", years=(2017, 2018, 2019, 2022))
    seven = ["ela_ach", "math_ach", "sci_ach"] + GAINS
    others = [c for c in COMPONENTS if c not in seven]
    df = df[(df.school_type == 1) & (df.n_components == 7) & df[seven].notna().all(axis=1)
            & df[others].isna().all(axis=1)].reset_index(drop=True)
    q = np.column_stack([df[["ela_ach", "math_ach", "sci_ach"]].mean(axis=1),
                         df[GAINS].mean(axis=1)])
    cut = _LEGACY[j] - 0.5
    w = (3 / 7, 4 / 7)
    X, names = _design(df)
    return RDDSample(
        Q=q, X=X, Y=df.y_pct_points.to_numpy(float), threshold=(cut, cut),
        name=f"fl_school_grades_two_score_{boundary}", feature_names=names,
        description=("Florida elementary schools with the seven-component model, 2017-2019 "
                     "and 2022. Q = (achievement mean, learning-gains mean); treatment = "
                     f"1{{(3 Q1 + 4 Q2)/7 >= {cut}}}; Y = next percent of points."),
        citation="Florida Department of Education, School Grades",
        treatment_rule=lambda Q: (Q @ np.array(w) >= cut).astype(int),
        extras={"weights": w, "cutoff": cut, "year": df.year.to_numpy(int),
                "eligible_published": df.grade.isin(list("ABCD"[:j + 1])).to_numpy(float)},
    )
