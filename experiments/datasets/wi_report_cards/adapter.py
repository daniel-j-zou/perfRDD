"""Wisconsin school report cards (DPI): a weighted sum of four priority-area scores whose
achievement/growth weights depend on the school's share of economically disadvantaged
students, then five rating bands.

Each Wisconsin public school gets an overall score from 0 to 100 and a star rating (Wis.
Stat. 115.385; DPI Report Card Technical Guide, "Building the Overall Score", every year
2015-16 to 2025-26). The priority-area scores, each on 0-100 and published to a tenth,
are Achievement (A), Growth (G), Closing Gaps (2015-16 to 2018-19) or Target Group
Outcomes (2020-21 on) (C), and On-Track to Graduation (O). With e the school's share of
economically disadvantaged (ED) students, as the fraction published in the file (three
decimals; the guides say whole percent, but only the three-decimal value reproduces the
published weights and scores),

    a(e) = 0.9 if e <= 0.05;  0.1 if e >= 0.65;  1 - (4/3 e + 1/30) in between,

and the multipliers are m_A = a(e), m_G = 1 - a(e) when G is present (m_A = 0.5 or 1 and
m_G = 0 when it is not), m_C = 0.5 when C is present (else 0). With f = 3 when the school
has a 3rd-grade ELA or 8th-grade mathematics On-Track component and f = 3.2 when not,

    w_K = m_K / (m_A + m_G + m_C) * f / 4   for K = A, G, C;    w_O = 1/4 (f = 3) or 1/5,
    x = w_A A + w_G G + w_C C + w_O O + ded,

where ded <= 0 is the Student Engagement Indicators deduction (0, -5, -10, ...; 2015-16 to
2018-19 only). The published overall score is round-half-up(x) to a tenth, and the
rating is a cutoff on it:

    rating year            5 stars   4 stars   3 stars   2 stars   (1 star below)
    2015-16 to 2018-19       83        73        63        53
    2020-21 to 2023-24       83        70        58        48
    2024-25                  84        71        60        49

(5 = Significantly Exceeds, 4 = Exceeds, 3 = Meets, 2 = Meets Few, 1 = Fails to Meet
Expectations). No report cards were produced for 2014-15 or 2019-20.

Q = x - (c - 0.05) for ``load()``: published overall >= c is exactly x >= c - 0.05
    (round half up), so D = 1{Q >= 0} is the higher rating. The rules allow 1e-9 of
    floating-point slack: x lands exactly on a rounding tie for 22-38 schools a year
    (most with e <= 0.05 or e >= 0.65, whose weights are on a 0.05 grid), and DPI
    rounds them up. D is a published star rating; the one statutory consequence tied to
    a school rating (Opportunity Schools and Partnership Program eligibility, 1 star)
    never applied in these years (README).
    ``load_two_score()``: Q = (A, G) and the rule w_i Q1 + (1 - w_i) Q2 >= c_i, with
    w_i = a(e_i), s_i = w_A + w_G and c_i = (c - 0.05 - offset_i) / s_i, offset_i =
    w_C C + w_O O + ded (per-unit weights and cutoffs in extras).
X = school characteristics that are not test scores: % economically disadvantaged (the
    weighting covariate), race/ethnicity shares, % with disabilities, % English
    learners, % open enrollment, log enrollment and its change from the prior year
    (WISEdash third-Friday counts), log district enrollment, lowest and highest grade,
    charter, virtual, independent (non-district) charter, school type, locale, CESA and
    year indicators.
Y = next report card year's published overall score, matched by district and school code.

Sample: public school report cards ("Public - All Students") with a published overall
score and one of the five ratings, rating years 2015-16, 2016-17, 2017-18, 2018-19 and
2020-21 to 2023-24, with Y observed. The outcome for 2018-19 is 2020-21. 2024-25 serves
only as an outcome. Private Choice-school report cards, alternate-accountability schools
(no score) and schools without a score are excluded.
"""
from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample
# Standard-library .xlsx reader (the environment has no openpyxl), as in tx_school_ratings.
from experiments.datasets.fl_school_grades._spreadsheet import read, sheet_names
from experiments.datasets.wi_report_cards.download import (ENR_YEARS, RC_YEARS,
                                                           district_file, enr_file, rc_file)

HERE = Path(__file__).parent
PROCESSED = HERE / "data" / "processed"
CACHE = PROCESSED / "wi_report_cards_panel.csv.gz"

YEARS = RC_YEARS                                       # fall year: 2017 = 2017-18
NEXT = dict(zip(YEARS[:-1], YEARS[1:]))                # next report card year
PREV = {v: k for k, v in NEXT.items()}
MAIN_YEARS = (2015, 2016, 2017, 2018, 2020, 2021, 2022, 2023)
WEIGHTS_PUBLISHED = (2017, 2018, 2020, 2021, 2022, 2023, 2024)   # weights in the file
# Minimum published overall score for 5, 4, 3 and 2 stars (Technical Guides; the
# 2024-25 cuts come from the 2025 standard setting and match the data exactly).
CUTOFFS = {y: (83.0, 73.0, 63.0, 53.0) for y in (2015, 2016, 2017, 2018)}
CUTOFFS.update({y: (83.0, 70.0, 58.0, 48.0) for y in (2020, 2021, 2022, 2023)})
CUTOFFS[2024] = (84.0, 71.0, 60.0, 49.0)
RATINGS = ["Fails to Meet Expectations", "Meets Few Expectations", "Meets Expectations",
           "Exceeds Expectations", "Significantly Exceeds Expectations"]
STARS = {r: i + 1 for i, r in enumerate(RATINGS)}
MAIN_STARS = 4      # Exceeds or better vs Meets or below: the most school-years near it
MPS = 3619          # Milwaukee Public Schools district code
EPS = 1e-9
TYPES = {"Elementary School": "elementary", "Middle School": "middle",
         "High School": "high", "Elementary/Secondary School": "combined",
         "Junior High School": "junior_high"}
_GRADE = {"K3": -2.0, "K4": -1.0, "PK": -1.0, "KG": 0.0}

# standardized name -> column names used across the years' files
_ALIAS = {
    "district": ["District Code"], "school": ["School Code"],
    "district_name": ["District Name"], "name": ["School Name"],
    "rctype": ["Report Card Type"], "agency": ["Agency type", "Agency Type"],
    "school_type": ["School Type"], "locale": ["Locale description"], "city": ["City"],
    "cesa": ["CESA"], "charter": ["Charter school indicator"],
    "virtual": ["Virtual school indicator"],
    "low_grade": ["Lowest Grade in the School", "Lowest Grade Level"],
    "high_grade": ["Highest Grade in the School", "Highest Grade Level"],
    "enrollment": ["School Enrollment"],
    "district_enrollment": ["District Enrollment", "District/LEA Enrollment"],
    "amind": ["Percent American Indian or Alaskan Native"], "asian": ["Percent Asian"],
    "black": ["Percent Black or African American"], "hispanic": ["Percent Hispanic/Latino"],
    "pacific": ["Percent Native Hawaiian or Other Pacific Islander"],
    "white": ["Percent White"], "multi": ["Percent Two or More Races"],
    "swd": ["Percent Students with Disabilities"], "econ": ["Percent Economically Disadvantaged"],
    "el": ["Percent Limited English Proficient", "Percent English Learners"],
    "open_enroll": ["Percent Open Enrollment"],
    "overall": ["Overall Accountability Score"], "rating_raw": ["Overall Accountability Rating"],
    "ach": ["School Student Achievement Score", "School Achievement Score", "Achievement Score"],
    "growth": ["School Student Growth Score", "School Growth Score", "Growth Score"],
    "gap": ["School Closing Gaps Score", "School Target Group Outcomes Score",
            "Target Group Outcomes Score"],
    "ontrack": ["School On-Track and Postsecondary Readiness Score",
                "School On-Track to Graduation Score", "On-Track to Graduation Score"],
    "ela3": ["School Third-Grade ELA Achievement Score", "School 3rd Grade ELA Achievement Score",
             "3rd Grade ELA Achievement Score"],
    "math8": ["School Eighth-Grade Mathematics Achievement Score",
              "School 8th Grade Mathematics Achievement Score",
              "8th Grade Mathematics Achievement Score"],
    "ded_raw": ["Student Engagement Indicators Deduction"],
    "w_ach_pub": ["Score weighting Achievement Priority Area"],
    "w_growth_pub": ["Score weighting Growth Priority Area"],
    "w_gap_pub": ["Score weighting Closing Gaps Priority Area",
                  "Score weighting Target Group Outcomes Priority Area"],
    "w_ontrack_pub": ["Score weighting Ontrack Priority Area",
                      "Score weighting On-Track Priority Area"],
}
_NUM = ["district", "school", "cesa", "enrollment", "district_enrollment", "amind", "asian",
        "black", "hispanic", "pacific", "white", "multi", "swd", "econ", "el", "open_enroll",
        "overall", "ach", "growth", "gap", "ontrack", "ela3", "math8", "w_ach_pub",
        "w_growth_pub", "w_gap_pub", "w_ontrack_pub"]
_TXT = ["district_name", "name", "rctype", "agency", "school_type", "locale", "city",
        "charter", "virtual", "rating_raw", "rating", "county", "y_rating", "weights_source"]


# ------------------------------------------------------------------------ raw files

def _need(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run python -m experiments.datasets.wi_report_cards.download")
    return path


def _grade(g) -> float:
    g = str(g).strip()
    if g in _GRADE:
        return _GRADE[g]
    try:
        return float(g)
    except ValueError:
        return np.nan


def _read_rc(year: int) -> pd.DataFrame:
    """One year's school report card data sheet (the last sheet), standardized."""
    path = _need(rc_file(year))
    rows = read(path, len(sheet_names(path)) - 1)
    hdr, seen = [], set()
    for h in rows[0]:
        h = str(h).strip()
        while h in seen:                       # two 'Notes' columns in 2015-16 to 2018-19
            h += "_"
        seen.add(h)
        hdr.append(h)
    raw = pd.DataFrame([list(r[:len(hdr)]) + [None] * (len(hdr) - len(r)) for r in rows[1:]],
                       columns=hdr)
    df = pd.DataFrame({"year": year}, index=raw.index)
    for k, names in _ALIAS.items():
        col = next((c for c in names if c in raw), None)
        df[k] = raw[col] if col else None
    for k in _NUM:
        df[k] = pd.to_numeric(df[k], errors="coerce")
    for k in ["district_name", "name", "rctype", "agency", "school_type", "locale", "city",
              "charter", "virtual", "rating_raw"]:
        df[k] = df[k].fillna("").astype(str).str.strip().replace({"NA": "", "None": ""})
    if year == 2015:                           # no Report Card Type column in 2015-16
        df["rctype"] = np.where(df.agency.isin(["Public school", "Non District Charter Schools"]),
                                "Public - All Students", "Private")
    # Ratings carry '^' (10-point change note, 2016-17 to 2018-19) or '*' (not rated).
    df["rating"] = df.rating_raw.str.rstrip("^*").str.strip()
    df["stars"] = df.rating.map(STARS)
    df["ded"] = pd.to_numeric(df.ded_raw.astype(str).str.extract(r"(-?\d+)")[0],
                              errors="coerce").fillna(0.0)
    df["low_grade"] = df.low_grade.map(_grade)
    df["high_grade"] = df.high_grade.map(_grade)
    return df.drop(columns="ded_raw").dropna(subset=["district", "school"])


def _read_enrollment(year: int) -> pd.DataFrame:
    """WISEdash certified enrollment (third Friday of September): school-level count of
    all students and the county."""
    with zipfile.ZipFile(_need(enr_file(year))) as z:
        member = next(n for n in z.namelist()
                      if n.lower().endswith(".csv") and "layout" not in n.lower())
        df = pd.read_csv(io.BytesIO(z.read(member)), dtype=str, encoding="latin-1",
                         usecols=["DISTRICT_CODE", "SCHOOL_CODE", "COUNTY", "GROUP_BY",
                                  "STUDENT_COUNT"])
    df = df[df.SCHOOL_CODE.notna() & (df.GROUP_BY == "All Students")]
    return pd.DataFrame({
        "year": year,
        "district": pd.to_numeric(df.DISTRICT_CODE, errors="coerce"),
        "school": pd.to_numeric(df.SCHOOL_CODE, errors="coerce"),
        "county": df.COUNTY.fillna("").str.strip(),
        "enr_wisedash": pd.to_numeric(df.STUDENT_COUNT, errors="coerce"),
    }).dropna(subset=["district", "school"]).drop_duplicates(["district", "school"])


# ---------------------------------------------------------------------------- rule

def ach_multiplier(econ) -> np.ndarray:
    """a(e): the achievement multiplier when Growth is present (e as a fraction)."""
    e = np.asarray(econ, float)
    return np.where(e <= 0.05, 0.9, np.where(e >= 0.65, 0.1, 1 - (4 / 3 * e + 1 / 30)))


def _weights(m_a, has_g, has_a, has_c, has38):
    m_a = np.where(has_g, m_a, 0.5)
    m_g = np.where(has_g, 1 - m_a, 0.0)
    m_a = np.where(has_a, m_a, 0.0)
    m_c = np.where(has_c, 0.5, 0.0)
    f = np.where(has38, 3.0, 3.2)
    tot = m_a + m_g + m_c
    with np.errstate(invalid="ignore", divide="ignore"):
        w = [m / tot * f / 4 for m in (m_a, m_g, m_c)]
    return w[0], w[1], w[2], np.where(has38, 0.25, 0.2)


def state_rule(df: pd.DataFrame) -> pd.DataFrame:
    """The Technical Guide's overall score from the published priority-area scores, with
    each step kept: the normalized achievement weight a (= m_A / (m_A + m_G) when Growth is
    present), the four weights, x (unrounded), and overall_computed = round-half-up(x).

    The weights come from the formula with the published % economically disadvantaged.
    Where that % is missing, or (2017-18 on) the formula's weights rounded to three
    decimals differ from the published ones, a is taken from the published weights,
    a = w_A / (w_A + w_G) (``weights_source`` = 'published')."""
    has_a, has_g = df.ach.notna(), df.growth.notna()
    has_c = df.gap.notna()
    has38 = df.ela3.notna() | df.math8.notna()
    a_formula = ach_multiplier(df.econ)
    wf = _weights(a_formula, has_g, has_a, has_c, has38)
    pub = [df.w_ach_pub, df.w_growth_pub, df.w_gap_pub, df.w_ontrack_pub]
    has_pub = df.w_ach_pub.notna()
    differ = np.zeros(len(df), bool)
    for w, p in zip(wf, pub):
        differ |= (p.notna() & (np.abs(np.round(w, 3) - p) > 1e-6)).to_numpy()
    a_pub = (df.w_ach_pub / (df.w_ach_pub + df.w_growth_pub)).to_numpy()
    use_pub = (has_pub & has_g & (df.econ.isna() | differ)).to_numpy()
    a = np.where(use_pub, a_pub, a_formula)
    w_a, w_g, w_c, w_o = (pd.Series(w, index=df.index)
                          for w in _weights(a, has_g, has_a, has_c, has38))
    offset = (w_c * df.gap).fillna(0) + w_o * df.ontrack + df.ded
    x = ((w_a * df.ach).fillna(0) + (w_g * df.growth).fillna(0) + offset).where(
        has_a & df.ontrack.notna())
    out = pd.DataFrame(index=df.index)
    out["a"] = np.where(has_g, a, np.nan)
    out["w_ach"], out["w_growth"], out["w_gap"], out["w_ontrack"] = w_a, w_g, w_c, w_o
    out["w_growth"] = out.w_growth.where(has_g)
    out["w_gap"] = out.w_gap.where(has_c)
    out["has38"] = has38.astype(float)
    out["scale"] = out.w_ach + out.w_growth                         # s_i (Growth present)
    out["offset"] = offset
    out["x"] = x
    out["overall_computed"] = np.floor(x * 10 + 0.5 + 1e-7) / 10
    out["x_tie"] = (np.abs(x * 10 - np.floor(x * 10) - 0.5) < 1e-7).astype(float)
    out["weights_source"] = np.where(use_pub, "published", "formula")
    out["weights_differ"] = differ.astype(float)
    return out


def _stars_from(score: pd.Series, year: pd.Series) -> pd.Series:
    cuts = np.array([CUTOFFS[int(y)] for y in year])
    s = score.to_numpy()[:, None]
    return pd.Series(1 + (s >= cuts[:, ::-1]).sum(axis=1), index=score.index).where(score.notna())


# --------------------------------------------------------------------------- panel

def _write_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.part")
    try:
        df.to_csv(tmp, index=False, compression="gzip")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def build() -> pd.DataFrame:
    """Every school report card row of 2015-16 to 2024-25 with a district code (private
    Choice schools have none and are dropped), with the standardized fields, the
    computed rule, WISEdash enrollment, the next report card year's outcomes and the
    previous year's overall score. Cached in ``data/processed`` (written atomically)."""
    if not CACHE.exists():
        _write_atomic(_build(), CACHE)
    p = pd.read_csv(CACHE, keep_default_na=False, na_values=[""], low_memory=False)
    for c in _TXT:
        if c in p:
            p[c] = p[c].fillna("").astype(str)
    return p


def _build() -> pd.DataFrame:
    p = pd.concat([_read_rc(y) for y in YEARS], ignore_index=True)
    p = pd.concat([p, state_rule(p)], axis=1)
    p["public"] = p.rctype.eq("Public - All Students").astype(float)
    p["scored"] = (p.overall.notna() & p.stars.notna()).astype(float)
    p["stars_computed"] = _stars_from(p.overall_computed, p.year)
    p["is_mps"] = p.district.eq(MPS).astype(float)
    p["independent_charter"] = p.agency.eq("Non District Charter Schools").astype(float)
    # Locale is missing for some schools in some years; it is fixed within a school.
    key = ["district", "school"]
    loc = p.locale.replace("", np.nan)
    p["locale"] = loc.groupby([p.district, p.school]).transform(
        lambda s: s.ffill().bfill()).fillna("")

    enr = pd.concat([_read_enrollment(y) for y in ENR_YEARS], ignore_index=True)
    p = p.merge(enr, on=["year"] + key, how="left")
    prev = enr[["year"] + key + ["enr_wisedash"]].rename(columns={"enr_wisedash": "enr_prev"})
    prev["year"] = prev.year + 1
    p = p.merge(prev, on=["year"] + key, how="left")
    p["county"] = p.county.fillna("")

    # Outcomes: the next report card year's public, scored row of the same school.
    pub = p[(p.public == 1) & (p.scored == 1)]
    nxt = pub[["year"] + key + ["overall", "stars", "rating", "ach", "growth", "x"]].rename(
        columns={"overall": "y_overall", "stars": "y_stars", "rating": "y_rating",
                 "ach": "y_ach", "growth": "y_growth", "x": "y_x"})
    nxt["year"] = nxt.year.map(PREV)
    nxt = nxt.dropna(subset=["year"]).astype({"year": int})
    p = p.merge(nxt, on=["year"] + key, how="left")
    p["y_year"] = p.year.map(NEXT)
    p["y_rating"] = p.y_rating.fillna("")
    lag = pub[["year"] + key + ["overall"]].rename(columns={"overall": "q_lag"})
    lag["year"] = lag.year.map(NEXT)
    lag = lag.dropna(subset=["year"]).astype({"year": int})
    p = p.merge(lag, on=["year"] + key, how="left")
    return p.sort_values(["year"] + key).reset_index(drop=True)


# -------------------------------------------------------------------------- checks

def district_ratings(codes=(MPS,)) -> pd.DataFrame:
    """Published district overall score and rating by year, from the district report
    card files (default: Milwaukee Public Schools, whose rating gates the Opportunity
    Schools and Partnership Program)."""
    rows = []
    for y in YEARS:
        path = _need(district_file(y))
        data = read(path, len(sheet_names(path)) - 1)
        hdr = [str(h).strip() for h in data[0]]
        i, j, k, r = (hdr.index(c) for c in ["District Code", "District Name",
                                              "Overall Accountability Score",
                                              "Overall Accountability Rating"])
        for row in data[1:]:
            row = list(row) + [None] * (len(hdr) - len(row))
            code = pd.to_numeric(row[i], errors="coerce")
            if code in codes:
                rows.append({"year": y, "district": int(code), "name": row[j],
                             "overall": row[k], "rating": str(row[r]).rstrip("^*")})
    return pd.DataFrame(rows)


def rule_check() -> pd.DataFrame:
    """Per report card year, public schools with a published overall score and rating:
    the computed overall score (round half up) against the published one, the computed
    weights against the published ones (2017-18 on), the published rating against the
    cut scores applied to the published score, and the stars implied by x."""
    p = build()
    rows = []
    for y, g in p.groupby("year"):
        m = (g.public == 1) & (g.scored == 1)
        h = g[m]
        r = {"year": int(y), "rows": len(g), "public_scored": int(m.sum()),
             "with_growth": int(h.growth.notna().sum()),
             "overall_agree": float((h.overall_computed == h.overall).mean()),
             "overall_miss": int((h.overall_computed != h.overall).sum()),
             "max_abs_diff": float((h.overall_computed - h.overall).abs().max()),
             "x_ties": int(h.x_tie.sum()),
             "weights_from_published": int((h.weights_source == "published").sum()),
             "rating_vs_published_score": float(
                 (_stars_from(h.overall, h.year) == h.stars).mean()),
             "stars_from_x_agree": float((h.stars_computed == h.stars).mean())}
        if y in WEIGHTS_PUBLISHED:
            ok = h.w_ach_pub.notna()
            r["weights_agree_3dp"] = float(
                (np.abs(h.w_ach.round(3) - h.w_ach_pub) < 1e-6)[ok].mean())
        rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------- samples

def _design(df: pd.DataFrame) -> tuple[np.ndarray, list]:
    X = pd.DataFrame(index=df.index)

    def fill(name, v, flag=None):
        """Median-impute; one missing-data indicator per block (``flag``)."""
        v = v.astype(float)
        X[name] = v.fillna(v.median())
        if flag and v.isna().any():
            X[flag] = np.maximum(X.get(flag, 0.0), v.isna().astype(float))

    # The school's demographic block (missing together, 2 school-years).
    fill("pct_econ_dis", 100 * df.econ, "demographics_missing")
    for c in ["black", "hispanic", "asian", "amind", "multi", "swd", "el"]:
        fill(f"pct_{c}", 100 * df[c], "demographics_missing")
    fill("log_enrollment", np.log(df.enrollment.where(df.enrollment > 0)),
         "demographics_missing")
    # District fields (missing for independent charters in 2023-24).
    fill("pct_open_enroll", 100 * df.open_enroll, "district_fields_missing")
    fill("log_district_enrollment",
         np.log(df.district_enrollment.where(df.district_enrollment > 0)),
         "district_fields_missing")
    fill("enrollment_change", np.log(df.enr_wisedash.where(df.enr_wisedash > 0)
                                     / df.enr_prev.where(df.enr_prev > 0)),
         "enrollment_change_missing")
    fill("low_grade", df.low_grade, "grades_missing")
    fill("high_grade", df.high_grade, "grades_missing")
    X["charter"] = df.charter.eq("Yes").astype(float)
    X["virtual"] = df.virtual.eq("Yes").astype(float)
    X["independent_charter"] = df.independent_charter.astype(float)
    for t in ["Middle School", "High School", "Elementary/Secondary School",
              "Junior High School"]:
        X[TYPES[t]] = (df.school_type == t).astype(float)
    for loc in ["City", "Suburb", "Town"]:                     # Rural is the base
        X[f"locale_{loc.lower()}"] = (df.locale == loc).astype(float)
    X["locale_missing"] = (df.locale == "").astype(float)
    for y in sorted(df.year.unique())[1:]:
        X[f"year_{y}"] = (df.year == y).astype(float)
    for c in sorted(df.cesa.dropna().unique()):
        if c != 1:                                            # CESA 1 (Milwaukee) is the base
            X[f"cesa_{int(c):02d}"] = (df.cesa == c).astype(float)
    X = X.loc[:, X.std() > 0]
    return X.to_numpy(float), list(X.columns)


def _frame(y_col: str = "y_overall", years=MAIN_YEARS, growth: bool = False) -> pd.DataFrame:
    p = build()
    m = p.year.isin(years) & (p.public == 1) & (p.scored == 1) & p.x.notna() & p[y_col].notna()
    if growth:
        m &= p.growth.notna() & p.ach.notna()
    return p[m].reset_index(drop=True)


def _cut(df: pd.DataFrame, stars: int) -> np.ndarray:
    """Per-row published cut score for `stars` or more."""
    return np.array([CUTOFFS[int(y)][5 - stars] for y in df.year])


def _extras(df: pd.DataFrame, stars: int) -> dict:
    t = df.school_type.map(TYPES).to_numpy()
    group = (df.year.astype(str) + "-" + t).to_numpy()
    cut = dict(zip(group, _cut(df, stars) - 0.05))
    return {
        "group": group, "group_cutoffs": {g: cut[g] for g in sorted(set(group))},
        "year": df.year.to_numpy(int), "school_type": t,
        "district": df.district.to_numpy(int), "school": df.school.to_numpy(int),
        "county": df.county.to_numpy(str), "cesa": df.cesa.to_numpy(float),
        "is_mps": df.is_mps.to_numpy(float),
        "ach": df.ach.to_numpy(float), "growth": df.growth.to_numpy(float),
        "gap": df.gap.to_numpy(float), "ontrack": df.ontrack.to_numpy(float),
        "deduction": df.ded.to_numpy(float), "econ": df.econ.to_numpy(float),
        "a": df.a.to_numpy(float),
        "w_ach": df.w_ach.to_numpy(float), "w_growth": df.w_growth.to_numpy(float),
        "w_gap": df.w_gap.to_numpy(float), "w_ontrack": df.w_ontrack.to_numpy(float),
        "weights_source": df.weights_source.to_numpy(str),
        "x": df.x.to_numpy(float), "overall_published": df.overall.to_numpy(float),
        "overall_computed": df.overall_computed.to_numpy(float),
        "rating_published": df.rating.to_numpy(str),
        "stars_published": df.stars.to_numpy(float),
        "eligible_published": (df.stars >= stars).to_numpy(float),
        "y_year": df.y_year.to_numpy(int), "q_lag": df.q_lag.to_numpy(float),
        "cutoff_table": CUTOFFS, "stars": stars,
    }


def _sample(stars: int = MAIN_STARS, y_col: str = "y_overall") -> RDDSample:
    df = _frame(y_col)
    X, names = _design(df)
    c = _cut(df, stars)
    q = (df.x - (c - 0.05)).to_numpy(float)
    ex = _extras(df, stars)
    lo, hi = RATINGS[stars - 2], RATINGS[stars - 1]
    return RDDSample(
        Q=q, X=X, Y=df[y_col].to_numpy(float), threshold=0.0,
        name=f"wi_report_cards_stars{stars}_{y_col}", feature_names=names,
        description=(f"Wisconsin school report cards, rating years "
                     f"{', '.join(f'{y}-{(y + 1) % 100:02d}' for y in MAIN_YEARS)}. "
                     "Q = x - (c - 0.05), x = w_A A + w_G G + w_C C + w_O O + deduction "
                     "(achievement/growth weights set by % economically disadvantaged), c the "
                     f"year's cut score; treatment = 1{{Q >= 0}} ({hi} or better rather than "
                     f"{lo}); Y = {y_col} (next report card year)."),
        citation="Wisconsin DPI, School Report Card Data Download Files 2015-16 to 2024-25, "
                 "and Report Card Technical Guides",
        treatment_rule=lambda Q: (Q >= -EPS).astype(int),
        extras=ex,
    )


def load() -> RDDSample:
    """4 stars (Exceeds Expectations) or better vs 3 or fewer; Y = next overall score."""
    return _sample(MAIN_STARS)


def load_stars5() -> RDDSample:
    """Significantly Exceeds vs Exceeds."""
    return _sample(5)


def load_stars4() -> RDDSample:
    return _sample(4)


def load_stars3() -> RDDSample:
    """Meets or better vs Meets Few or Fails to Meet."""
    return _sample(3)


def load_stars2() -> RDDSample:
    """Meets Few or better vs Fails to Meet: the boundary with a statutory consequence
    (Opportunity Schools and Partnership Program eligibility for MPS schools), which never
    applied in these years because MPS was never rated Fails to Meet (README)."""
    return _sample(2)


def load_two_score(stars: int = MAIN_STARS) -> RDDSample:
    """Q = (Achievement, Growth) for schools with both; treatment = 1{w_i Q1 + (1 - w_i) Q2
    >= c_i} with per-unit weights and cutoffs (extras["unit_weights"], ["unit_cutoff"]).

    w_i = a(e_i), the normalized achievement weight; c_i = (c - 0.05 - offset_i) / s_i with
    s_i = w_A + w_G and offset_i = w_C C + w_O O + deduction. ``extras["weights"]`` and
    ``extras["cutoff"]`` hold a fixed reference rule (median w_i, median c_i) for scripts
    that need one; it is not the deployed rule. Same X and Y as ``load()``."""
    df = _frame(growth=True)
    X, names = _design(df)
    c = _cut(df, stars)
    w = df.a.to_numpy(float)
    s = df.scale.to_numpy(float)
    unit_cut = (c - 0.05 - df.offset.to_numpy(float)) / s
    n = len(df)

    def rule(Q):
        Q = np.asarray(Q, float)
        if Q.shape != (n, 2):
            raise ValueError(f"the deployed rule has per-unit weights for the {n} loaded "
                             f"rows; got Q of shape {Q.shape}")
        return (w * Q[:, 0] + (1 - w) * Q[:, 1] >= unit_cut - EPS).astype(int)

    ex = _extras(df, stars)
    w_ref = round(float(np.median(w)), 6)
    ex.update({
        "unit_weights": np.column_stack([w, 1 - w]), "unit_cutoff": unit_cut,
        "scale": s, "offset": df.offset.to_numpy(float),
        "cutoff_overall": c - 0.05,
        "weights": (w_ref, round(1 - w_ref, 6)), "cutoff": float(np.median(unit_cut)),
        "reference_rule_note": "weights/cutoff are medians of the per-unit values, "
                               "not the deployed rule",
    })
    return RDDSample(
        Q=df[["ach", "growth"]].to_numpy(float), X=X, Y=df.y_overall.to_numpy(float),
        threshold=(0.0, 0.0), name=f"wi_report_cards_two_score_stars{stars}",
        feature_names=names,
        description=("Wisconsin school report cards. Q = (Achievement, Growth); treatment = "
                     "1{w_i Q1 + (1 - w_i) Q2 >= c_i}, w_i = 0.9 - 0.8 * clip((e_i - 0.05) "
                     "/ 0.6, 0, 1) from % economically disadvantaged e_i, c_i the cut score "
                     "net of the other priority areas and deductions; Y = next report card "
                     "year's overall score."),
        citation="Wisconsin DPI, School Report Card Data Download Files and Technical Guides",
        treatment_rule=rule, extras=ex,
    )
