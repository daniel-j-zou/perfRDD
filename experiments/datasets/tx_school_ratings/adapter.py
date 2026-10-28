"""Texas A-F campus ratings (TEA): 70% of the better of two domains plus 30% of a third.

TEA rates each campus on three domains, each a scaled score from 0 to 100: Student
Achievement (SA), School Progress (SP, itself the better of Part A: Academic Growth and
Part B: Relative Performance) and Closing the Gaps (CtG). Chapter 5 of each year's
Accountability Manual gives the overall scaled score as

    M = max(SA, SP), capped at 89 if either is below 60,
    overall = round-half-up(0.7 M + 0.3 CtG)                 (= M if CtG is missing),

then two caps on the result: at most 59 if three of SA, Growth, Relative Performance and
CtG are below 60 ("3 Fs"; from 2019 only when SA < 60), and from 2023 at most 69 if three
of them are below 70 and SA < 70 ("3 Ds"). The overall grade is A >= 90, B 80-89,
C 70-79, D 60-69, F < 60 in every year. Exceptions: 2018 campuses were rated Met Standard
(>= 60) or Improvement Required; 2022 campuses below 70 were "Not Rated: Senate Bill
1365"; AEA campuses can add bonus points (2018-2022); paired campuses carry their
partner's overall score and have no domain scores; in districts with a local
accountability system (LAS) the published overall blends in a local score.

Q = x - (c - 0.5), with x = 0.7 M + 0.3 CtG the unrounded overall score and c the
    published cutoff of the chosen boundary. The domain scores are whole numbers, so x
    lies on a 0.1 grid; because the published overall is round-half-up(x), "published
    overall >= c" is exactly "x >= c - 0.5", and D = 1{Q >= 0} = the higher grade.
    ``load_two_score()``: Q = (M, CtG) with weights (0.7, 0.3). ``load_three_score()``:
    Q = (SA, SP, CtG) with the max rule.
X = campus characteristics that do not depend on that year's tests: % economically
    disadvantaged and % emergent bilingual (PEIMS October snapshot), log number of
    students, charter, school type, lowest and highest grade served, the district's log
    enrollment and % economically disadvantaged, ESC region and year indicators.
Y = load(): the next rated year's overall scaled score, matched by campus number (in LAS
    district-years, the state score recomputed from the published domains). Other
    loaders: next SA, next CtG, no row next rated year.

Sample: non-AEA campuses with all three domain scores and a published overall score in
2019, 2022, 2023, 2024 and 2025, outside LAS district-years, with Y observed. 2018 (no
campus letter grades) is used only for the lagged score; 2026 only as the outcome for
2025. 2020 and 2021 were not rated (COVID-19 disaster declaration), so the outcome for
2019 is 2022. The 2023 ratings were published on 2025-04-24 and the 2024 ratings on
2025-08-15 (litigation), after the next year's tests; ``extras["released_after_outcome"]``
flags them.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample
# The FLDOE reader is a standard-library .xlsx reader; the environment has no openpyxl.
from experiments.datasets.fl_school_grades._spreadsheet import read, sheet_names
from experiments.datasets.tx_school_ratings.download import RATINGS, RAW

HERE = Path(__file__).parent
PROCESSED = HERE / "data" / "processed"
CACHE = PROCESSED / "tx_school_ratings_panel.csv.gz"

YEARS = (2018, 2019, 2022, 2023, 2024, 2025, 2026)      # rating years with scores
NEXT = dict(zip(YEARS[:-1], YEARS[1:]))                 # next rated year
PREV = {v: k for k, v in NEXT.items()}
MAIN_YEARS = (2019, 2022, 2023, 2024, 2025)
RELEASED_AFTER_OUTCOME = (2023, 2024)   # published 2025-04-24 and 2025-08-15
WEIGHTS = (0.7, 0.3)
# Minimum published overall scaled score for A, B, C, D (F below D). The same in every
# year, 2018-2026 (Chapter 5, "Cut Scores for Scaling Conversion"); 2018 campuses got
# Met Standard at >= 60, and 2022 campuses below 70 got "Not Rated: Senate Bill 1365".
CUTOFFS = {y: (90, 80, 70, 60) for y in YEARS}
BOUNDARIES = {"AB": 0, "BC": 1, "CD": 2, "DF": 3}
MAIN_BOUNDARY = "AB"   # the most campus-years within 2 points of the cutoff (README)
GRADE_POINTS = {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0, "F": 0.0}
SB1365 = "Not Rated: Senate Bill 1365"
TYPES = {"Elementary": "elementary", "Middle School": "middle", "High School": "high",
         "Elem/Secondary": "combined"}
_GRADE = {"EE": -2, "PK": -1, "KG": 0}

_COLS = {
    "District Number": "district", "District": "district_name", "Campus Number": "campus",
    "Campus": "name", "Region": "region", "County": "county", "School Type": "school_type",
    "Grades Served": "grades", "Alternative Education Accountability": "aea",
    "Charter": "charter", "Number of Students": "n_students",
    "% Economically Disadvantaged": "econ", "% EB/EL Students": "eb", "% LEP Students": "eb",
    "Overall Rating": "rating", "Overall Score": "overall",
    "Student Achievement Score": "sa", "School Progress Score": "sp",
    "Academic Growth Score": "growth", "Relative Performance Score": "relperf",
    "Closing the Gaps Score": "ctg", "Paired Type": "paired",
    "Campus Enrollment Type": "enrollment_type",
}
SCORES = ["overall", "sa", "sp", "growth", "relperf", "ctg"]


# ------------------------------------------------------------------------ raw file

def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.strip(), errors="coerce")


def _str(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().replace({"None": "", "nan": ""})


def _grade(g: str) -> float:
    g = g.strip()
    return float(_GRADE[g]) if g in _GRADE else (float(g) if g.isdigit() else np.nan)


def _read_year(path: Path, sheet: int, year: int) -> pd.DataFrame:
    rows = read(path, sheet)
    h = next(k for k, r in enumerate(rows) if r and str(r[0] or "").startswith("District"))
    hdr = [" ".join(str(x).split()).replace(" *", "").replace("Grade/Rating", "Rating")
           if x is not None else f"col{j}" for j, x in enumerate(rows[h])]
    df = pd.DataFrame([list(r[:len(hdr)]) + [None] * (len(hdr) - len(r)) for r in rows[h + 1:]],
                      columns=hdr)
    df = df[[c for c in hdr if c in _COLS]].rename(columns=_COLS)
    out = pd.DataFrame({"year": year}, index=df.index)
    for c in ["district", "district_name", "campus", "name", "county", "school_type", "grades",
              "aea", "charter", "rating", "paired", "enrollment_type"]:
        out[c] = _str(df[c]) if c in df else ""
    out["region"] = df.region.astype(str).str.extract(r"REGION (\d+)", expand=False).astype(float)
    for c in ["n_students", "econ", "eb"] + SCORES:
        out[c] = _num(df[c]) if c in df else np.nan
    out = out[out.district != ""]
    span = out.grades.str.split("-", n=1, expand=True).reindex(columns=[0, 1]).fillna("")
    out["low_grade"] = span[0].map(_grade)
    out["high_grade"] = span[1].map(_grade)
    return out.reset_index(drop=True)


def _read_all() -> pd.DataFrame:
    path = RAW / RATINGS
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run python -m experiments.datasets.tx_school_ratings.download")
    names = sheet_names(path)
    frames = []
    for y in YEARS:
        i = names.index(f"{y} State Summary")
        frames.append(_read_year(path, i, y))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------- rule

def state_rule(df: pd.DataFrame) -> pd.DataFrame:
    """The Chapter 5 overall score from the published domain scores, with each step kept:
    m (the better of SA and SP, after the 89 cap), x (unrounded 0.7 m + 0.3 CtG),
    overall_formula (round half up), the 3 Fs / 3 Ds caps, and overall_computed."""
    sa, sp, ctg = df.sa, df.sp, df.ctg
    m = np.fmax(sa, sp)
    m_cap = (np.fmin(sa, sp) < 60) & (m > 89)
    m = m.where(~m_cap, 89.0)
    ten = 7 * m + 3 * ctg                       # 10 x, an integer
    out = pd.DataFrame(index=df.index)
    out["m"] = m
    out["m_cap_binds"] = m_cap.astype(float)
    out["x"] = ten / 10
    formula = np.floor((ten + 5) / 10)
    four = df[["sa", "growth", "relperf", "ctg"]]
    allfour = four.notna().all(axis=1)
    f3 = allfour & ((four < 60).sum(axis=1) >= 3) & ((sa < 60) | (df.year == 2018))
    d3 = allfour & ((four < 70).sum(axis=1) >= 3) & (sa < 70) & (df.year >= 2023)
    comp = formula.where(~f3, np.minimum(formula, 59))
    comp = comp.where(~d3, np.minimum(comp, 69))
    out["overall_formula"] = formula
    out["cap_3f"] = (f3 & (formula > 59)).astype(float)
    out["cap_3d"] = (d3 & (formula > 69)).astype(float)
    out["overall_computed"] = comp
    return out


def _expected_rating(score: pd.Series, year: pd.Series) -> pd.Series:
    letter = pd.Series(np.select([score >= 90, score >= 80, score >= 70, score >= 60],
                                 list("ABCD"), "F"), index=score.index)
    letter = letter.where(~((year == 2022) & (score < 70)), SB1365)
    letter = letter.where(year != 2018, np.where(score >= 60, "Met Standard",
                                                 "Improvement Required"))
    return letter.where(score.notna(), "")


def _valid_rating(rating: pd.Series, year: pd.Series) -> pd.Series:
    """The published rating is an ordinary rating label for that year (not Not Rated,
    an appeal note or a data-integrity flag)."""
    ok = rating.isin(list("ABCDF")) & ~year.isin([2018, 2022])
    ok |= rating.isin(["A", "B", "C", SB1365]) & (year == 2022)
    ok |= rating.isin(["Met Standard", "Improvement Required"]) & (year == 2018)
    return ok


# --------------------------------------------------------------------------- panel

def build() -> pd.DataFrame:
    """Every campus-year of 2018, 2019 and 2022-2026 with the published scores, the
    computed rule, district context, flags, the next rated year's outcomes and the
    previous rated year's score."""
    if not CACHE.exists():
        df = _build()
        PROCESSED.mkdir(parents=True, exist_ok=True)
        df.to_csv(CACHE, index=False)
    p = pd.read_csv(CACHE, keep_default_na=False, na_values=[""], low_memory=False,
                    dtype={"district": str, "campus": str})
    for c in ["district_name", "name", "county", "school_type", "grades", "aea", "charter",
              "rating", "paired", "enrollment_type", "y_rating"]:
        p[c] = p[c].fillna("").astype(str)
    return p


def _build() -> pd.DataFrame:
    raw = _read_all()
    dist = raw[raw.campus == ""][["year", "district", "n_students", "econ"]].rename(
        columns={"n_students": "district_n_students", "econ": "district_econ"})
    p = raw[raw.campus != ""].copy()
    p = p.merge(dist, on=["year", "district"], how="left")
    p = pd.concat([p, state_rule(p)], axis=1)
    p["three_domains"] = p[["sa", "sp", "ctg"]].notna().all(axis=1).astype(float)
    p["is_aea"] = p.aea.eq("Yes").astype(float)
    # Paired campuses (named in 2023+; before 2023 they show an overall score but no
    # domain scores) carry another campus's or the district's score.
    p["is_paired"] = ((p.paired != "") | (p.overall.notna() & p.sa.isna())).astype(float)
    # LAS district-years: a non-AEA campus whose published overall differs from the
    # Chapter 5 state score (Chapter 11: LAS blends a local score into campuses rated C or
    # better). Found for Dallas ISD 2019 and 2022, Snyder ISD 2019, Bellville ISD 2023-2026.
    chk = (p.is_aea == 0) & (p.three_domains == 1) & p.overall.notna()
    bad = chk & (p.overall_computed != p.overall)
    las = p[bad].groupby(["year", "district"]).size().rename("n_las_mismatch").reset_index()
    p = p.merge(las, on=["year", "district"], how="left")
    p["las_district"] = p.n_las_mismatch.notna().astype(float)
    p["n_las_mismatch"] = p.n_las_mismatch.fillna(0)
    p["valid_rating"] = _valid_rating(p.rating, p.year).astype(float)
    # The state overall score: published, except in LAS district-years, where it is
    # recomputed from the published domains (when they exist).
    p["overall_state"] = p.overall.where(
        ~((p.las_district == 1) & p.overall_computed.notna()), p.overall_computed)
    p["released_after_outcome"] = p.year.isin(RELEASED_AFTER_OUTCOME).astype(float)

    # Next rated year, matched by campus number.
    key = ["campus"]
    p["y_year"] = p.year.map(NEXT)
    nxt = p[key + ["year", "overall_state", "x", "sa", "ctg", "rating", "is_paired",
                   "is_aea"]].rename(columns={
        "year": "y_year", "overall_state": "y_overall", "x": "y_x", "sa": "y_sa",
        "ctg": "y_ctg", "rating": "y_rating", "is_paired": "y_paired", "is_aea": "y_aea"})
    p = p.merge(nxt, on=key + ["y_year"], how="left", indicator=True)
    p["y_closed"] = np.where(p.y_year.notna(), (p._merge == "left_only").astype(float), np.nan)
    p = p.drop(columns="_merge")
    # A paired campus's published overall is its partner's, not its own outcome.
    p.loc[p.y_paired == 1, "y_overall"] = np.nan
    p["y_rating"] = p.y_rating.fillna("")
    p["y_grade_points"] = p.y_rating.map(GRADE_POINTS)
    # Previous rated year's state overall score (not in X: it depends on that year's grade).
    lag = p[key + ["year", "overall_state", "x"]].rename(
        columns={"overall_state": "q_lag", "x": "x_lag"})
    lag["year"] = lag.year.map(NEXT)
    lag = lag.dropna(subset=["year"]).astype({"year": int})
    p = p.merge(lag, on=key + ["year"], how="left")
    return p.sort_values(["year", "campus"]).reset_index(drop=True)


# -------------------------------------------------------------------------- checks

def rule_check() -> pd.DataFrame:
    """Per rating year: the published overall score against the Chapter 5 rule computed
    from the published domain scores (non-AEA campuses with all three domains), before
    and after dropping LAS district-years, and the published rating against the cut
    points applied to the published overall score."""
    p = build()
    rows = []
    for y, g in p.groupby("year"):
        base = (g.is_aea == 0) & (g.three_domains == 1) & g.overall.notna()
        keep = base & (g.las_district == 0)
        rated = g.overall.notna() & (g.valid_rating == 1) & (g.is_aea == 0)
        aea = (g.is_aea == 1) & (g.three_domains == 1) & g.overall.notna()
        rows.append({
            "year": y, "campuses": len(g), "aea": int((g.is_aea == 1).sum()),
            "paired": int((g.is_paired == 1).sum()),
            "no_overall": int(g.overall.isna().sum()),
            "nonaea_three_domains": int(base.sum()),
            "agree_all": float((g.overall_computed == g.overall)[base].mean()),
            "las_districts": int(g[g.las_district == 1].district.nunique()),
            "las_campuses": int((base & (g.las_district == 1)).sum()),
            "agree_excl_las": float((g.overall_computed == g.overall)[keep].mean()),
            "agree_70_30_only": float((g.overall_formula == g.overall)[keep].mean()),
            "cap_3f": int(g.cap_3f[keep].sum()), "cap_3d": int(g.cap_3d[keep].sum()),
            "cap_89": int(g.m_cap_binds[keep].sum()),
            "rating_agree": float((_expected_rating(g.overall, g.year) == g.rating)[rated].mean()),
            "rating_rows": int(rated.sum()),
            "aea_agree": float((g.overall_computed == g.overall)[aea].mean()),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------- samples

def _design(df: pd.DataFrame) -> tuple[np.ndarray, list]:
    X = pd.DataFrame(index=df.index)

    def fill(name, v):
        X[name] = v.fillna(v.median())
        if v.isna().any():
            X[f"{name}_missing"] = v.isna().astype(float)

    fill("pct_econ_dis", 100 * df.econ)
    fill("pct_eb", 100 * df.eb)
    fill("log_students", np.log(df.n_students.where(df.n_students > 0)))
    fill("district_log_students", np.log(df.district_n_students.where(df.district_n_students > 0)))
    fill("district_pct_econ_dis", 100 * df.district_econ)
    fill("low_grade", df.low_grade)
    fill("high_grade", df.high_grade)
    X["charter"] = df.charter.eq("Yes").astype(float)
    for t in ["Middle School", "High School", "Elem/Secondary"]:
        X[TYPES[t]] = (df.school_type == t).astype(float)
    for y in sorted(df.year.unique())[1:]:
        X[f"year_{y}"] = (df.year == y).astype(float)
    for r in sorted(df.region.dropna().unique()):
        if r != 4:                                   # Region 4 (Houston) is the base
            X[f"region_{int(r):02d}"] = (df.region == r).astype(float)
    X = X.loc[:, X.std() > 0]
    return X.to_numpy(float), list(X.columns)


def _frame(y_col: str = "y_overall", years=MAIN_YEARS) -> pd.DataFrame:
    p = build()
    m = (p.year.isin(years) & (p.is_aea == 0) & (p.three_domains == 1) & p.overall.notna()
         & (p.las_district == 0) & (p.valid_rating == 1) & p[y_col].notna())
    return p[m].reset_index(drop=True)


def _eligible(df: pd.DataFrame, j: int) -> np.ndarray:
    """1 if the published rating is the higher grade of boundary j or better. In 2022,
    D and F were both published as Not Rated: SB 1365, so the D/F boundary uses the
    published overall score instead."""
    higher = set("ABCD"[:j + 1])
    out = df.rating.isin(higher).to_numpy(float)
    c = CUTOFFS[2022][j]
    if j == 3:
        m22 = (df.year == 2022).to_numpy()
        out[m22] = (df.overall[m22] >= c).to_numpy(float)
    return out


def _extras(df: pd.DataFrame, j: int, cut: float) -> dict:
    t = df.school_type.map(TYPES).to_numpy()
    group = (df.year.astype(str) + "-" + t).to_numpy()
    return {
        "group": group, "group_cutoffs": {g: cut for g in sorted(set(group))},
        "year": df.year.to_numpy(int), "school_type": t,
        "district": df.district.to_numpy(str), "campus": df.campus.to_numpy(str),
        "region": df.region.to_numpy(float),
        "sa": df.sa.to_numpy(float), "sp": df.sp.to_numpy(float),
        "ctg": df.ctg.to_numpy(float), "m": df.m.to_numpy(float),
        "growth": df.growth.to_numpy(float), "relperf": df.relperf.to_numpy(float),
        "x": df.x.to_numpy(float), "overall_published": df.overall.to_numpy(float),
        "rating_published": df.rating.to_numpy(str),
        "eligible_published": _eligible(df, j),
        "cap_3f": df.cap_3f.to_numpy(float), "cap_3d": df.cap_3d.to_numpy(float),
        "y_year": df.y_year.to_numpy(int), "q_lag": df.q_lag.to_numpy(float),
        "released_after_outcome": df.released_after_outcome.to_numpy(float),
        "sb1365_year": (df.year == 2022).to_numpy(float),
        "cutoff_table": CUTOFFS,
    }


def _sample(boundary: str = MAIN_BOUNDARY, y_col: str = "y_overall") -> RDDSample:
    j = BOUNDARIES[boundary]
    c = CUTOFFS[MAIN_YEARS[0]][j]
    df = _frame(y_col)
    X, names = _design(df)
    hi, lo = "ABCD"[j], "BCDF"[j]
    # Q = x - (c - 0.5) = (7 m + 3 CtG + 5 - 10 c) / 10, exact on the 0.1 grid.
    q = (7 * df.m + 3 * df.ctg + 5 - 10 * c).to_numpy(float) / 10
    ex = _extras(df, j, c - 0.5)
    ex["boundary"] = boundary
    return RDDSample(
        Q=q, X=X, Y=df[y_col].to_numpy(float), threshold=0.0,
        name=f"tx_school_ratings_{boundary}_{y_col}", feature_names=names,
        description=(f"Texas A-F campus ratings, {', '.join(map(str, MAIN_YEARS))}. "
                     f"Q = 0.7 max(SA, SP) + 0.3 CtG - ({c} - 0.5); treatment = 1{{Q >= 0}} "
                     f"(overall {hi} or better rather than {lo}); Y = {y_col} "
                     "(next rated year)."),
        citation="Texas Education Agency, 2026 Statewide Multi-Year Ratings Spreadsheet "
                 "and Accountability Manuals 2018-2026, Chapter 5",
        treatment_rule=lambda Q: (Q >= 0).astype(int),
        extras=ex,
    )


def load() -> RDDSample:
    """A/B boundary; Y = next rated year's overall scaled score."""
    return _sample(MAIN_BOUNDARY)


def load_ab() -> RDDSample:
    return _sample("AB")


def load_bc() -> RDDSample:
    return _sample("BC")


def load_cd() -> RDDSample:
    return _sample("CD")


def load_df() -> RDDSample:
    return _sample("DF")


def load_next_achievement() -> RDDSample:
    return _sample(MAIN_BOUNDARY, "y_sa")


def load_next_ctg() -> RDDSample:
    return _sample(MAIN_BOUNDARY, "y_ctg")


def load_closed() -> RDDSample:
    """Y = 1 if the campus has no row in the next rated year's file."""
    return _sample(MAIN_BOUNDARY, "y_closed")


def load_two_score(boundary: str = MAIN_BOUNDARY) -> RDDSample:
    """Q = (M, CtG), M = max(SA, SP) after the 89 cap; treatment = 1{0.7 M + 0.3 CtG >=
    c - 0.5}. Same sample, X and Y as ``load()``."""
    j = BOUNDARIES[boundary]
    cut = CUTOFFS[MAIN_YEARS[0]][j] - 0.5
    df = _frame()
    X, names = _design(df)
    w = np.array(WEIGHTS)
    ex = _extras(df, j, cut)
    ex.update({"weights": WEIGHTS, "cutoff": cut, "boundary": boundary})
    return RDDSample(
        Q=df[["m", "ctg"]].to_numpy(float), X=X, Y=df.y_overall.to_numpy(float),
        threshold=(cut, cut), name=f"tx_school_ratings_two_score_{boundary}",
        feature_names=names,
        description=("Texas A-F campus ratings. Q = (max(SA, SP), CtG); treatment = "
                     f"1{{0.7 Q1 + 0.3 Q2 >= {cut}}}; Y = next rated year's overall score."),
        citation="Texas Education Agency, 2026 Statewide Multi-Year Ratings Spreadsheet",
        # Tolerance: 0.7 Q1 + 0.3 Q2 lies on a 0.1 grid and equals the cutoff for many
        # campuses, which floating point can put a hair below it.
        treatment_rule=lambda Q: (Q @ w >= cut - 1e-9).astype(int),
        extras=ex,
    )


def load_three_score(boundary: str = MAIN_BOUNDARY) -> RDDSample:
    """Q = (SA, SP, CtG); treatment = 1{0.7 M + 0.3 CtG >= c - 0.5} with M = max(SA, SP),
    capped at 89 if min(SA, SP) < 60 (the cap never binds in the sample)."""
    j = BOUNDARIES[boundary]
    cut = CUTOFFS[MAIN_YEARS[0]][j] - 0.5
    df = _frame()
    X, names = _design(df)

    def rule(Q):
        m = np.maximum(Q[:, 0], Q[:, 1])
        m = np.where(np.minimum(Q[:, 0], Q[:, 1]) < 60, np.minimum(m, 89), m)
        return (0.7 * m + 0.3 * Q[:, 2] >= cut - 1e-9).astype(int)

    ex = _extras(df, j, cut)
    ex.update({"cutoff": cut, "boundary": boundary, "rule": "0.7 max(Q1, Q2) + 0.3 Q3"})
    return RDDSample(
        Q=df[["sa", "sp", "ctg"]].to_numpy(float), X=X, Y=df.y_overall.to_numpy(float),
        threshold=(cut, cut, cut), name=f"tx_school_ratings_three_score_{boundary}",
        feature_names=names,
        description=("Texas A-F campus ratings. Q = (SA, SP, CtG); treatment = "
                     f"1{{0.7 max(Q1, Q2) + 0.3 Q3 >= {cut}}}; Y = next rated year's "
                     "overall score."),
        citation="Texas Education Agency, 2026 Statewide Multi-Year Ratings Spreadsheet",
        treatment_rule=rule, extras=ex,
    )
