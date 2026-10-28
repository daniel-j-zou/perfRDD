"""North Carolina School Performance Grades (NC DPI): a fixed-weight sum of two scores
with A-F bands, and the low-performing designation.

Each North Carolina public school gets a School Performance Grade (SPG) from two scores
on 0-100 (G.S. 115C-83.15; DPI Technical Guide 2025-26, section 2.1):

* the achievement score A, a composite percent of all achievement indicators, rounded
  to a tenth;
* the growth score G, the EVAAS composite growth index converted to 100 points, rounded
  to a tenth. Status "Exceeded" is index >= 2.0, which is G >= 85.0 in every row.

SPG score = round-half-up(0.8 A + 0.2 G) to a whole number (A alone when G is missing),
and the grade is A >= 85, B >= 70, C >= 55, D >= 40, F below (15-point scale, every year
2013-14 to 2024-25). A school is low-performing when its grade is D or F and its growth
status is "Met" or "Not Met" (G.S. 115C-105.37(a)).

Q = (A, G) in score units. Because the published SPG score is round-half-up of
    0.8 A + 0.2 G, "published score >= c" is exactly "0.8 A + 0.2 G >= c - 0.5": the
    cutoffs on the unrounded index are 84.5, 69.5, 54.5 and 39.5. A and G are on a
    0.1 grid, so the index is on a 0.02 grid and lands exactly on a cutoff for some
    schools; the treatment rules allow 1e-6 of floating-point slack, and ties round up,
    as DPI rounds them.
load():  D = 1{0.8 Q1 + 0.2 Q2 >= 69.5}, a grade of B or better rather than C (the B/C
    boundary has the most schools near it, narrowly ahead of C/D).
load_low_performing(): D = 1{0.8 Q1 + 0.2 Q2 < 54.5} * 1{Q2 < 85}, the published
    low-performing designation. Not low-performing is the union of the two half-planes;
    the growth arm binds only below the kink at (Q1, Q2) = (46.875, 85).
X = school characteristics not computed from test scores: % economically disadvantaged;
    Black, Hispanic, Asian, American Indian and multiracial shares, English-learner and
    disability shares (shares of the year's test takers); log final average daily
    membership and its change from the prior year; % inexperienced teachers; charter,
    Title I, calendar, school type and level; county and year indicators.
Y = next grading year's published SPG score, matched by the six-character school code.
    Other loaders: next achievement, next growth, next low-performing designation, closure.

Sample: school-years graded A-F in 2017-18, 2018-19, 2021-22, 2022-23 and 2023-24
(spring years 2018, 2019, 2022-2024) with both scores and a next-year SPG score.
Excluded: K-2 schools that inherit another school's SPG; alternative schools in the
modified accountability model (no grade); schools without a growth score (their SPG is
achievement alone, so they are off the two-score rule); masked grades. 2019-20 and
2020-21 have no grades (COVID-19), so the outcome for 2018-19 comes from 2021-22. 2024-25
is graded but has no outcome yet. 2013-14 to 2016-17 are in the cache but not in the
loaders: their published achievement score is a whole number, so the rule cannot be
reproduced from the published scores (``rule_check()``).
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample
# Standard-library .xlsx reader (the environment has no openpyxl); shared with the Florida
# dataset rather than copied.
from experiments.datasets.fl_school_grades._spreadsheet import read_xlsx
from experiments.datasets.nc_school_grades.download import SRC1, SRC2

HERE = Path(__file__).parent
PROCESSED = HERE / "data" / "processed"
CACHE = PROCESSED / "nc_school_grades_panel.csv.gz"

WEIGHTS = (0.8, 0.2)
GRADE_MIN = {"A": 85, "B": 70, "C": 55, "D": 40}      # G.S. 115C-83.15(d), every year
BOUNDARIES = {"AB": 85, "BC": 70, "CD": 55, "DF": 40}
MAIN_BOUNDARY = "BC"   # the most school-years within 2 points of the cutoff (README)
GROWTH_EXCEEDED = 85.0  # growth index >= 2.0 <=> growth score >= 85.0
GRADED_YEARS = (2014, 2015, 2016, 2017, 2018, 2019, 2022, 2023, 2024, 2025)
MAIN_YEARS = (2018, 2019, 2022, 2023, 2024)
NEXT = dict(zip(GRADED_YEARS[:-1], GRADED_YEARS[1:]))
PREV = {v: k for k, v in NEXT.items()}
GRADE_POINTS = {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0, "F": 0.0}
EPS = 1e-6
_TXT = ["agency_code", "name", "lea_code", "county", "designation", "category",
        "school_type", "calendar", "title_i", "asm_option", "k2_feeder", "grade_raw",
        "grade", "eg_status", "lp", "rlp", "clpc", "y_grade", "y_lp"]


# ------------------------------------------------------------------------ raw files

def _need(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run python -m experiments.datasets.nc_school_grades.download")
    return path


def _xlsx(zpath: Path, table: str) -> pd.DataFrame:
    """One SRC table (first row = header) read from inside the zip, without extracting."""
    with zipfile.ZipFile(_need(zpath)) as z:
        member = next(n for n in z.namelist() if n.endswith(f"/{table}.xlsx"))
        rows = read_xlsx(io.BytesIO(z.read(member)), 0)
    hdr = [str(h).strip().lower() for h in rows[0]]
    df = pd.DataFrame([r + [None] * (len(hdr) - len(r)) for r in rows[1:]], columns=hdr)
    df = df[df.year.notna()].copy()
    df["year"] = pd.to_numeric(df.year).astype(int)
    df["agency_code"] = df.agency_code.astype(str).str.strip()
    return df


def _schools(df: pd.DataFrame) -> pd.DataFrame:
    """School rows only (drops district 'xxxLEA' and state 'NC-SEA' rows)."""
    a = df.agency_code
    return df[(a.str.len() == 6) & ~a.str.endswith("LEA") & (a != "NC-SEA")]


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _str(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip()


def _spg() -> pd.DataFrame:
    """Overall (subgroup ALL) SPG rows: rcd_acc_spg1 for 2014-2017, spg2 for 2018 on."""
    out = []
    for table in ("rcd_acc_spg1", "rcd_acc_spg2"):
        d = _schools(_xlsx(SRC1, table))
        d = d[_str(d.subgroup) == "ALL"]
        out.append(pd.DataFrame({
            "year": d.year, "agency_code": d.agency_code,
            "asm_option": _str(d.asm_option),
            "k2_feeder": _str(d.k2_feeder) if "k2_feeder" in d else "",
            "ach": _num(d.ach_score), "growth": _num(d.eg_score),
            "spg_pub": _num(d.spg_score), "grade_raw": _str(d.spg_grade),
            "eg_status": _str(d.eg_status),
            "grade_masked": (_num(d.spg_grade_masking).notna() if "spg_grade_masking" in d
                             else False),
        }))
    p = pd.concat(out, ignore_index=True)
    # 'A+NG' (A with no significant achievement gaps, 2015-2017) is an A.
    p["grade"] = p.grade_raw.str.replace("+NG", "", regex=False)
    return p


def _growth_index() -> pd.DataFrame:
    d = _schools(_xlsx(SRC1, "rcd_acc_eg"))
    d = d[(_str(d.subject) == "ALL") & (_str(d.subgroup) == "ALL")]
    return pd.DataFrame({"year": d.year, "agency_code": d.agency_code,
                         "eg_index": _num(d.eg_index)}).drop_duplicates(["year", "agency_code"])


def _lowperf() -> pd.DataFrame:
    d = _schools(_xlsx(SRC1, "rcd_acc_lowperf"))
    return pd.DataFrame({"year": d.year, "agency_code": d.agency_code, "lp": _str(d.lp_school),
                         "rlp": _str(d.rlp_school), "clpc": _str(d.clpc_school)})


def _eds() -> pd.DataFrame:
    d = _schools(_xlsx(SRC1, "rcd_acc_eds"))
    m = _num(d.pct_eds_masking)
    # Masking codes: 1 = above 95%, 2 = below 5%, 3 = fewer than 10 students.
    v = _num(d.pct_eds).where(~m.isin([1, 2, 3]))
    v = v.mask(m == 1, 97.5).mask(m == 2, 2.5)
    return pd.DataFrame({"year": d.year, "agency_code": d.agency_code, "pct_eds": v})


def _location() -> pd.DataFrame:
    d = _schools(_xlsx(SRC2, "rcd_location"))
    d = d[_str(d.agency_level) == "SCH"]
    return pd.DataFrame({
        "year": d.year, "agency_code": d.agency_code, "name": _str(d["name"]),
        "lea_code": _str(d.lea_code), "county": _str(d.county),
        "designation": _str(d.designation_type), "category": _str(d.category_code),
        "school_type": _str(d.school_type), "calendar": _str(d.calendar_type),
        "title_i": _str(d.title_i), "grade_span": _str(d.grade_span),
    }).drop_duplicates(["year", "agency_code"])


def _adm() -> pd.DataFrame:
    d = _schools(_xlsx(SRC1, "rcd_adm"))
    d = d[_str(d.category_code) == ""]
    return pd.DataFrame({"year": d.year, "agency_code": d.agency_code,
                         "adm": _num(d.avg_student_num)}).drop_duplicates(["year", "agency_code"])


def _teachers() -> pd.DataFrame:
    """% inexperienced teachers: 0-3 years of experience (rcd_experience, to 2018) and
    beginning teachers (rcd_eq, 2019 on). The definition changes with the table."""
    x = _schools(_xlsx(SRC2, "rcd_experience"))
    x = x[_str(x.staff) == "Teacher"]
    x = pd.DataFrame({"year": x.year, "agency_code": x.agency_code,
                      "pct_inexp_teachers": 100 * _num(x.pct_experience_0)})
    q = _schools(_xlsx(SRC2, "rcd_eq"))
    q = pd.DataFrame({"year": q.year, "agency_code": q.agency_code,
                      "pct_inexp_teachers": 100 * _num(q.pct_beg_teachers)})
    t = pd.concat([x[x.year <= 2018], q[q.year >= 2019]], ignore_index=True)
    return t.dropna().drop_duplicates(["year", "agency_code"])


SHARES = {"BL7": "share_black", "HI7": "share_hispanic", "AS7": "share_asian",
          "AM7": "share_amind", "MU7": "share_multi", "WH7": "share_white",
          "ELS": "share_els", "SWD": "share_swd", "AIG": "share_aig"}


def _test_counts() -> pd.DataFrame:
    """Subgroup shares of the year's test scores (rcd_acc_pc, subject and grade ALL).
    Only the counts are used, never the scores. A subgroup with fewer than 10 scores has
    no row, and its share is set to 0."""
    keep = []
    with zipfile.ZipFile(_need(SRC1)) as z:
        member = next(n for n in z.namelist() if n.endswith("/rcd_acc_pc.txt"))
        with z.open(member) as fh:
            for ch in pd.read_csv(fh, sep="\t", dtype=str, chunksize=2_000_000,
                                  usecols=["year", "agency_code", "subject", "grade",
                                           "subgroup", "den"]):
                keep.append(ch[(ch.subject == "ALL") & (ch.grade == "ALL")
                               & ch.subgroup.isin(list(SHARES) + ["ALL"])])
    d = pd.concat(keep, ignore_index=True)
    d["year"] = d.year.astype(int)
    d["den"] = _num(d.den)
    d = d.drop_duplicates(["year", "agency_code", "subgroup"])
    w = d.pivot(index=["year", "agency_code"], columns="subgroup", values="den")
    out = pd.DataFrame(index=w.index)
    out["n_test_scores"] = w["ALL"]
    for k, v in SHARES.items():
        out[v] = 100 * w[k].fillna(0) / w["ALL"]
    return out.reset_index()


# -------------------------------------------------------------------------- panel

def build() -> pd.DataFrame:
    """Every school with an overall SPG row, 2013-14 to 2024-25, with the computed
    scores and grades, covariates, lagged scores and next-grading-year outcomes."""
    if not CACHE.exists():
        df = _build()
        PROCESSED.mkdir(parents=True, exist_ok=True)
        df.to_csv(CACHE, index=False)
    p = pd.read_csv(CACHE, keep_default_na=False, na_values=[""],
                    dtype={c: str for c in _TXT})
    for c in _TXT:
        p[c] = p[c].fillna("")
    return p


def _build() -> pd.DataFrame:
    p = _spg()
    key = ["year", "agency_code"]
    loc, adm = _location(), _adm()
    for t in (_growth_index(), _lowperf(), _eds(), loc, adm, _teachers(), _test_counts()):
        p = p.merge(t, on=key, how="left")
    for c in _TXT:
        if c in p:
            p[c] = p[c].fillna("").astype(str)

    # The rule, in integer hundredths: 100 x (0.8 A + 0.2 G) = 8 A10 + 2 G10.
    a10 = np.round(p.ach * 10)
    g10 = np.round(p.growth * 10)
    both = p.ach.notna() & p.growth.notna()
    p["index"] = np.where(both, (8 * a10 + 2 * g10) / 100, p.ach)
    idx100 = np.where(both, 8 * a10 + 2 * g10, 100 * p.ach)
    p["spg_computed"] = np.where(p.ach.notna(), np.floor((idx100 + 50) / 100), np.nan)
    p["grade_from_pub"] = _grade(p.spg_pub)
    p["grade_computed"] = _grade(p.spg_computed)
    p["lp_computed"] = np.where(
        both, ((p["index"] < 54.5 - EPS) & (p.growth < GROWTH_EXCEEDED - EPS)).astype(float),
        np.nan)
    p["graded"] = p.grade.isin(list("ABCDF")) & ~p.grade_masked.astype(bool)

    # Lagged scores (previous grading year) and next-grading-year outcomes.
    sk = ["agency_code", "year"]
    lag = p[sk + ["ach", "growth", "spg_pub"]].rename(
        columns={"ach": "ach_lag", "growth": "growth_lag", "spg_pub": "spg_lag"})
    lag["year"] = lag.year.map(NEXT)
    p = p.merge(lag.dropna(subset=["year"]).astype({"year": int}), on=sk, how="left")
    nxt = p[sk + ["ach", "growth", "spg_pub", "grade", "lp", "graded"]].rename(
        columns={"year": "y_year", "ach": "y_ach", "growth": "y_growth", "spg_pub": "y_spg",
                 "grade": "y_grade", "lp": "y_lp", "graded": "y_graded"})
    p["y_year"] = p.year.map(NEXT)
    p = p.merge(nxt, on=["agency_code", "y_year"], how="left")
    p.loc[p.y_graded != True, "y_spg"] = np.nan          # noqa: E712 (NaN-safe)
    p["y_grade_points"] = p.y_grade.map(GRADE_POINTS)
    p["y_lp"] = p.y_lp.fillna("")
    p["y_lp_flag"] = np.where(p.y_lp.isin(["Y", "N"]), (p.y_lp == "Y").astype(float), np.nan)

    # Calendar-year neighbours: enrollment in the prior and next school year, closure.
    loc = loc[key]
    for k, name in ((-1, "adm_prev"), (1, "adm_next")):
        a = adm.rename(columns={"adm": name}).assign(year=lambda d: d.year - k)
        p = p.merge(a, on=key, how="left")
    last = loc.year.max()
    seen = set(zip(loc.year, loc.agency_code))
    p["y_closed"] = [float((y + 1, c) not in seen) if y + 1 <= last else np.nan
                     for y, c in zip(p.year, p.agency_code)]
    p["y_log_adm_change"] = np.log(p.adm_next.where(p.adm_next > 0)) - np.log(
        p.adm.where(p.adm > 0))
    return p.drop(columns=["y_graded"]).sort_values(key).reset_index(drop=True)


def _grade(score: pd.Series) -> np.ndarray:
    s = np.asarray(score, float)
    g = np.select([s >= 85, s >= 70, s >= 55, s >= 40], list("ABCD"), "F").astype(object)
    g[np.isnan(s)] = ""
    return g


# --------------------------------------------------------------------------- checks

def rule_check() -> pd.DataFrame:
    """By year: the published grade against the 15-point scale applied to the published
    SPG score (every graded school); the published SPG score and grade against the rule
    applied to the published A and G (two-score schools); growth status against G; and
    the published low-performing flag against the statutory rule and against the
    two-cutoff rule in (A, G)."""
    p = build()
    p = p[p.graded & (p.k2_feeder != "Y")]
    rows = []
    for y, g in p.groupby("year"):
        two = g[g.ach.notna() & g.growth.notna()]
        lp_known = g.lp.isin(["Y", "N"])
        ex = two.eg_status.eq("Exceeded")
        rows.append({
            "year": y, "graded": len(g), "no_growth": int(g.growth.isna().sum()),
            "two_score": len(two),
            "grade_vs_pub_score": float((g.grade == g.grade_from_pub).mean()),
            "score_rule": float((two.spg_computed == two.spg_pub).mean()),
            # 2014-2017 publish A as a whole number: is the published score reachable
            # from some A in [A - 0.5, A + 0.5]?
            "score_within_rounding": float(
                ((two.spg_pub >= np.floor(0.8 * (two.ach - 0.5) + 0.2 * two.growth + 0.5))
                 & (two.spg_pub <= np.floor(0.8 * (two.ach + 0.5) + 0.2 * two.growth + 0.5))
                 ).mean()),
            "grade_rule": float((two.grade_computed == two.grade).mean()),
            "status_vs_G": float(((two.growth >= GROWTH_EXCEEDED) == ex).mean()),
            "lp_published": int((g.lp == "Y").sum()) if lp_known.any() else None,
            "lp_statute": (float(((g.grade.isin(["D", "F"])
                                   & g.eg_status.isin(["Met", "NotMet"])) == (g.lp == "Y"))
                                 [lp_known].mean()) if lp_known.any() else None),
            "lp_score_rule": (float((two.lp_computed == (two.lp == "Y"))[two.lp.isin(["Y", "N"])]
                                    .mean()) if lp_known.any() else None),
        })
    return pd.DataFrame(rows)


def boundary_mass(width: float = 2.0, years=MAIN_YEARS) -> dict:
    """School-years in the loaders' sample within `width` points of each index cutoff."""
    df = _frame("y_spg", years)
    return {b: int((np.abs(df["index"] - (c - 0.5)) < width).sum())
            for b, c in BOUNDARIES.items()}


# -------------------------------------------------------------------------- samples

def _design(df: pd.DataFrame) -> tuple[np.ndarray, list]:
    X = pd.DataFrame(index=df.index)
    X["pct_eds"] = df.pct_eds.fillna(df.pct_eds.median())
    X["eds_missing"] = df.pct_eds.isna().astype(float)
    for c in ["share_black", "share_hispanic", "share_asian", "share_amind", "share_multi",
              "share_els", "share_swd"]:
        X[c] = df[c].fillna(df[c].median())
    X["shares_missing"] = df.share_black.isna().astype(float)
    la = np.log(df.adm.where(df.adm > 0))
    X["log_adm"] = la.fillna(la.median())
    X["adm_missing"] = la.isna().astype(float)
    ch = la - np.log(df.adm_prev.where(df.adm_prev > 0))
    X["adm_change"] = ch.fillna(0.0).clip(-1, 1)
    X["adm_change_missing"] = ch.isna().astype(float)
    X["pct_inexp_teachers"] = df.pct_inexp_teachers.fillna(df.pct_inexp_teachers.median())
    X["teachers_missing"] = df.pct_inexp_teachers.isna().astype(float)
    X["charter"] = (df.designation == "C").astype(float)
    X["title_i"] = (df.title_i == "Y").astype(float)
    cal = df.calendar.str.lower()
    X["year_round"] = cal.str.contains("year").astype(float)
    X["modified_calendar"] = cal.str.contains("modified").astype(float)
    st = df.school_type.str.lower()
    X["alternative"] = st.str.contains("alternative").astype(float)
    X["magnet"] = st.str.contains("magnet").astype(float)
    X["special_school"] = st.str.contains("exceptional|special|ec school|hospital|"
                                          "vocational|cte").astype(float)
    for c, name in [("M", "middle"), ("H", "high"), ("I", "elem_middle"),
                    ("T", "middle_high"), ("A", "k12")]:
        X[name] = (df.category == c).astype(float)       # elementary is the base
    dummies = {f"year_{y}": (df.year == y).astype(float) for y in sorted(df.year.unique())[1:]}
    county = df.county.replace("", "Unknown")
    dummies.update({f"county_{c.replace(' ', '_')}": (county == c).astype(float)
                    for c in sorted(county.unique()) if c != "Wake"})  # Wake is the base
    X = pd.concat([X, pd.DataFrame(dummies, index=df.index)], axis=1)
    X = X.loc[:, X.std() > 0]
    return X.to_numpy(float), list(X.columns)


def _frame(y_col: str, years=MAIN_YEARS) -> pd.DataFrame:
    p = build()
    df = p[p.year.isin(years) & p.graded & (p.k2_feeder != "Y") & (p.asm_option != "Y")
           & p.ach.notna() & p.growth.notna() & p[y_col].notna()]
    return df.reset_index(drop=True)


def _index_at_least(cut: float, w=WEIGHTS):
    w = np.asarray(w, float)
    return lambda Q: (Q @ w >= cut - EPS).astype(int)


def _low_performing_rule(cut: float = 54.5, g_cut: float = GROWTH_EXCEEDED, w=WEIGHTS):
    w = np.asarray(w, float)
    return lambda Q: ((Q @ w < cut - EPS) & (Q[:, 1] < g_cut - EPS)).astype(int)


def _extras(df: pd.DataFrame) -> dict:
    return {
        "weights": WEIGHTS, "year": df.year.to_numpy(int),
        "agency_code": df.agency_code.to_numpy(str), "county": df.county.to_numpy(str),
        "lea_code": df.lea_code.to_numpy(str), "category": df.category.to_numpy(str),
        "index": df["index"].to_numpy(float), "spg_published": df.spg_pub.to_numpy(float),
        "grade_published": df.grade.to_numpy(str), "growth_status": df.eg_status.to_numpy(str),
        "growth_index": df.eg_index.to_numpy(float),
        "lp_published": (df.lp == "Y").to_numpy(float),
        "y_year": df.y_year.to_numpy(int),
        "q_lag": df[["ach_lag", "growth_lag"]].to_numpy(float),
        "spg_lag": df.spg_lag.to_numpy(float),
        "share_aig": df.share_aig.to_numpy(float),
    }


def _sample(boundary: str = MAIN_BOUNDARY, y_col: str = "y_spg") -> RDDSample:
    c = BOUNDARIES[boundary]
    cut = c - 0.5
    hi = "ABCD"[list(BOUNDARIES).index(boundary)]
    df = _frame(y_col)
    X, names = _design(df)
    return RDDSample(
        Q=df[["ach", "growth"]].to_numpy(float), X=X, Y=df[y_col].to_numpy(float),
        threshold=(cut, cut), name=f"nc_school_grades_{boundary}_{y_col}",
        feature_names=names,
        description=(f"North Carolina School Performance Grades, spring years "
                     f"{', '.join(map(str, MAIN_YEARS))}. Q = (achievement score, growth "
                     f"score); treatment = 1{{0.8 Q1 + 0.2 Q2 >= {cut}}} (grade {hi} or "
                     f"better); Y = {y_col} (next grading year)."),
        citation="NC Department of Public Instruction, School Report Card data set "
                 "2024-25 (rcd_acc_spg1/2, rcd_acc_lowperf and companion tables)",
        treatment_rule=_index_at_least(cut),
        extras={**_extras(df), "cutoff": cut, "boundary": boundary,
                "eligible_published": df.grade.isin(list("ABCD"[:"ABCD".index(hi) + 1]))
                .to_numpy(float)},
    )


def load() -> RDDSample:
    """B/C boundary: D = 1{0.8 achievement + 0.2 growth >= 69.5}; Y = next SPG score."""
    return _sample(MAIN_BOUNDARY)


def load_ab() -> RDDSample:
    return _sample("AB")


def load_bc() -> RDDSample:
    return _sample("BC")


def load_cd() -> RDDSample:
    return _sample("CD")


def load_df() -> RDDSample:
    return _sample("DF")


def load_next_ach() -> RDDSample:
    return _sample(MAIN_BOUNDARY, "y_ach")


def load_next_growth() -> RDDSample:
    return _sample(MAIN_BOUNDARY, "y_growth")


def load_closed() -> RDDSample:
    """Y = 1 if the school is not in the next school year's directory (rcd_location)."""
    return _sample(MAIN_BOUNDARY, "y_closed")


def load_low_performing(y_col: str = "y_spg") -> RDDSample:
    """Low-performing designation (G.S. 115C-105.37): grade D or F and growth not
    exceeded, i.e. D = 1{0.8 Q1 + 0.2 Q2 < 54.5} * 1{Q2 < 85}. D = 1 is the designation
    (the policy treatment); extras["eligible_published"] is DPI's published flag."""
    df = _frame(y_col)
    X, names = _design(df)
    return RDDSample(
        Q=df[["ach", "growth"]].to_numpy(float), X=X, Y=df[y_col].to_numpy(float),
        threshold=(54.5, GROWTH_EXCEEDED), name=f"nc_school_grades_low_performing_{y_col}",
        feature_names=names,
        description=("North Carolina low-performing schools, spring years "
                     f"{', '.join(map(str, MAIN_YEARS))}. Q = (achievement, growth); "
                     "treatment = 1{0.8 Q1 + 0.2 Q2 < 54.5} * 1{Q2 < 85}; "
                     f"Y = {y_col} (next grading year)."),
        citation="NC Department of Public Instruction, School Report Card data set "
                 "2024-25 (rcd_acc_spg2, rcd_acc_lowperf)",
        treatment_rule=_low_performing_rule(),
        extras={**_extras(df), "cutoff": 54.5, "growth_cutoff": GROWTH_EXCEEDED,
                "eligible_published": (df.lp == "Y").to_numpy(float)},
    )


def load_low_performing_next_lp() -> RDDSample:
    """Low-performing rule; Y = 1 if designated low-performing next grading year."""
    return load_low_performing("y_lp_flag")
