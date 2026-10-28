"""Louisiana school performance scores (LDOE): A-F letter grades from cutoffs on the SPS.

Each Louisiana public school gets a School Performance Score (SPS, 0-150) that is a
fixed-weight sum of component indices (each 0-150, published to one decimal), and a
letter grade from fixed cutoffs on the SPS: A >= 90, B >= 75, C >= 60, D >= 50, F below,
in every graded year 2017-18 to 2024-25. The weights depend on the school's grades:

* K-8, no grade 8, 2017-18 and 2018-19: 75% assessment + 25% progress;
* K-8 with grade 8: 70% assessment + 25% progress + 5% dropout/credit accumulation (DCAI);
* from 2020-21, 5% interests & opportunities (I&O) is added and the assessment weight
  falls by 5 points (70/25/0/5 and 65/25/5/5);
* high schools: 12.5% assessment, 12.5% progress, 25% ACT/WorkKeys, 25% cohort
  graduation rate index, 25% strength of diploma; from 2020-21 the graduation rate
  index has 20% and I&O 5%;
* combination schools: K-8 and high-school SPS weighted by their numbers of testers.

LDOE's SPS calculators round each index to 0.1 and the weighted sum to 0.1 (half up);
the grade compares that rounded SPS with the cutoffs.

Q = load(): the published SPS minus (own cutoff - 0.05). "published SPS >= c" is exactly
    "unrounded SPS >= c - 0.05", so D = 1{Q >= 0} = the higher of the two grades. For the
    C/D and D/F boundaries the policy treatment (CIR, state intervention) is 1 - D.
X = school characteristics from the October 1 enrollment count of the same school year
    (before the spring tests): % economically disadvantaged, race/ethnicity shares, %
    female, % limited English proficient, log enrollment, grade-span shares, charter,
    selective admissions, school type, year and parish indicators.
Y = load(): next SPS year's published SPS, matched by site code. Other loaders: next
    assessment index, next grade (A=4 ... F=0), and no SPS next year.

Sample: school-years graded A-F on the regular ("School" or "Closed School") sheets in
2017-18, 2018-19, 2021-22, 2022-23 and 2023-24 (spring years 2018, 2019, 2022-2024).
Excluded: "T" (turnaround) grades; alternative schools (separate formula and sheet);
K-2 schools with no tested grades, whose scores come from the schools they feed;
2024-25 (no next-year SPS under this formula). 2019-20 has no SPS (COVID-19); 2020-21 has simulated
SPS without grades, used only as the outcome for 2018-19. ``extras["y_year"]`` records
the outcome year.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample
from experiments.datasets.fl_school_grades._spreadsheet import read, sheet_names
from experiments.datasets.la_school_grades.download import ENROLLMENT, RAW, SPS

HERE = Path(__file__).parent
PROCESSED = HERE / "data" / "processed"
CACHE = PROCESSED / "la_school_grades_panel.csv.gz"

TYPES = {1: "elementary_middle", 2: "high", 3: "combination"}
_TYPE_CODE = {"Elementary/Middle School": 1, "High School": 2, "Combination School": 3}
GRADED_YEARS = (2018, 2019, 2022, 2023, 2024, 2025)
# Minimum published SPS for A, B, C, D (F below D), by spring year. Sources: the
# 2019-2020 SPS overview (2017-18 to 2020-21) and the 2023 SPS release FAQ (2022-23);
# for 2023-24 and 2024-25 the same scale is used in LDOE's 2022-23 calculator and is
# confirmed by the files (``rule_check()``). The rise to 95/80/65/50 (2021-22) and
# 100/85/70/50 (2024-25) planned in the 2019-20 overview did not take effect.
CUTOFFS = {y: (90.0, 75.0, 60.0, 50.0) for y in GRADED_YEARS}
BOUNDARIES = {"AB": 0, "BC": 1, "CD": 2, "DF": 3}
MAIN_BOUNDARY = "BC"   # the most school-years within 2 points of the cutoff (README)
MAIN_YEARS = (2018, 2019, 2022, 2023, 2024)
NEXT = {2018: 2019, 2019: 2021, 2021: 2022, 2022: 2023, 2023: 2024, 2024: 2025}
GRADE_POINTS = {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0, "F": 0.0}
HALF = 0.05            # half of the 0.1 rounding unit of the published SPS

# Component columns of the SPS files: (output name, header prefix after the year).
_COMP = [
    ("ai", "K8 & High School Assessment Index"), ("pi", "K8 & High School Progress Index"),
    ("k8_ai", "K8 Assessment Index"), ("k8_pi", "K8 Progress Index"),
    ("dcai", "Dropout Credit Accumulation Index"), ("k8_io", "K8 Interests & Opportunities"),
    ("hs_ai", "High School Assessment Index"), ("hs_pi", "High School Progress Index"),
    ("act", "ACT Index"), ("hs_io", "High School Interests & Opportunities"),
]
# K-8 weights (assessment, progress, DCAI, I&O) in thousandths; LDOE 2021-22 and
# 2022-23 SPS calculators. Before 2020-21 there is no I&O index.
K8_WEIGHTS = {
    "old": {False: (750, 250, 0, 0), True: (700, 250, 50, 0)},      # keyed by has DCAI
    "io": {False: (700, 250, 0, 50), True: (650, 250, 50, 50)},
}


# ------------------------------------------------------------------------ raw files

def _num(s: pd.Series) -> pd.Series:
    # 2020-21 simulated scores carry a "C19-" prefix (e.g. "C19-60.5").
    t = s.astype(str).str.strip().str.replace(r"^C19-", "", regex=True)
    return pd.to_numeric(t.replace({"": None, "None": None}), errors="coerce")


def _txt(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().replace({"None": ""})


def _sheet(path: Path, i: int) -> pd.DataFrame:
    rows = read(path, i)
    h = next(k for k, r in enumerate(rows) if r and r[0] == "Site Code")
    hdr = [" ".join(str(x).split()) if x is not None else f"col{j}"
           for j, x in enumerate(rows[h])]
    body = []
    for r in rows[h + 1:]:
        if r and r[0] == "Site Code":       # a second table (closed alternative schools)
            break
        if r and isinstance(r[0], str) and re.fullmatch(r"[0-9A-Z]{6}", r[0].strip()):
            body.append(r + [None] * (len(hdr) - len(r)))
    df = pd.DataFrame(body, columns=hdr)
    return df.loc[:, ~df.columns.duplicated()]      # 2018: first block = new formula


def _col(df: pd.DataFrame, prefix: str) -> pd.Series:
    hit = [c for c in df.columns if c.startswith(prefix)]
    return df[hit[0]] if hit else pd.Series([None] * len(df), index=df.index)


def _read_year(year: int) -> pd.DataFrame:
    path = RAW / "sps" / SPS[year]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run python -m experiments.datasets.la_school_grades.download")
    out = []
    for i, name in enumerate(sheet_names(path)):
        if "Alt" in name:
            continue
        df = _sheet(path, i)
        y = str(year)
        res = pd.DataFrame({
            "year": year, "code": _txt(df["Site Code"]), "name": _txt(df["School"]),
            "system": _txt(df["School System"]),
            "school_type": _txt(_col(df, "School Type")).map(_TYPE_CODE),
            "grade": _txt(_col(df, f"{y} Letter Grade")),
            "sps": _num(_col(df, f"{y} SPS") if year != 2021
                        else _col(df, f"{y} Simulated SPS")),
            "closed_sheet": float("Closed" in name),
            "aus": _txt(_col(df, f"{y} Academically Unacceptable")),
            # "Magnet School" = selective admissions. In 2017-18 the column also marks
            # alternative and juvenile schools, which later have their own sheet.
            "selective": _txt(_col(df, f"{y} Selective Admissions")).eq("Magnet School")
                         .astype(float),
            "alternative": _txt(_col(df, f"{y} Selective Admissions")).isin(
                ["Alternative School", "Juvenile School"]).astype(float),
        })
        for k, p in _COMP:
            raw = _txt(_col(df, f"{y} {p}"))
            res[k] = _num(raw)
            res[f"{k}_masked"] = (raw.ne("") & res[k].isna()).astype(float)
            res[f"{k}_present"] = raw.ne("").astype(float)
        res["sod"] = _num(_col(df, "Strength of Diploma"))
        res["cgr_index"] = _num(_col(df, "Cohort Graduation Rate Index"))
        out.append(res)
    return pd.concat(out, ignore_index=True)


def _pct(s: pd.Series) -> pd.Series:
    t = s.astype(str).str.strip()
    v = pd.to_numeric(t.str.rstrip("%"), errors="coerce")
    return np.where(t.str.endswith("%"), v / 100, v)


def _enrollment(year: int) -> pd.DataFrame:
    """October 1 "Multiple Statistics by Site" for the school year ending in `year`."""
    path = RAW / "enrollment" / ENROLLMENT[year][1]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run python -m experiments.datasets.la_school_grades.download")
    i = next(k for k, n in enumerate(sheet_names(path)) if "Site" in n)
    rows = read(path, i)
    head = [" ".join(str(x).split()) for r in rows[:8] for x in r if x is not None]
    ext = any("Extension Academy" in h for h in head)
    data = [r for r in rows[3:] if r and isinstance(r[2], str)
            and re.fullmatch(r"[0-9A-Z]{6}", r[2].strip())]
    grades = ["inf", "presch", "prek", "k"] + [f"g{g}" for g in range(1, 10)] + ["gt9"] + \
             [f"g{g}" for g in (10, 11, 12)] + (["ext"] if ext else [])
    names = (["sys", "sysname", "code", "fed", "site", "total", "female", "male",
              "amind", "asian", "black", "hispanic", "hawpi", "white", "multi", "minority",
              "fep", "lep"] + grades + ["ed", "nonprofit", "charter_type", "rollup",
                                        "parish"])
    df = pd.DataFrame([r[:len(names)] for r in data], columns=names)
    res = pd.DataFrame({"year": year, "code": df.code.str.strip(),
                        "enrollment": pd.to_numeric(df.total, errors="coerce")})
    for c in ["female", "lep", "ed"]:
        res[f"pct_{c}"] = _pct(df[c])
    n = res.enrollment.where(res.enrollment > 0)
    for c in ["black", "hispanic", "asian", "white"]:
        res[f"pct_{c}"] = pd.to_numeric(df[c], errors="coerce") / n
    cnt = {g: pd.to_numeric(df[g], errors="coerce").fillna(0) for g in grades}
    res["share_prek_2"] = sum(cnt[g] for g in ["inf", "presch", "prek", "k", "g1", "g2"]) / n
    res["share_3_5"] = sum(cnt[g] for g in ["g3", "g4", "g5"]) / n
    res["share_6_8"] = sum(cnt[g] for g in ["g6", "g7", "g8"]) / n
    res["share_9_12"] = sum(cnt[g] for g in ["g9", "gt9", "g10", "g11", "g12"]) / n
    res["has_grade8"] = (cnt["g8"] > 0).astype(float)
    res["tested_grades"] = sum(cnt[g] for g in ["g3", "g4", "g5", "g6", "g7", "g8", "g9",
                                                 "gt9", "g10", "g11", "g12"])
    res["charter"] = df.charter_type.fillna("").astype(str).str.strip().replace(
        {"None": ""}).ne("").astype(float)
    res["parish"] = pd.to_numeric(df.parish, errors="coerce")
    return res[~res.code.duplicated(keep=False)]


def _round1(tot_e4: np.ndarray) -> np.ndarray:
    """LDOE's ROUND(x, 1), half up, on a sum held in units of 1e-4 (exact integers)."""
    return np.floor((tot_e4 + 500) / 1000) / 10


def k8_formula(p: pd.DataFrame) -> pd.Series:
    """SPS of a K-8 school from its published indices; NaN where an index is masked or
    the progress index is missing. 2017-18 and 2018-19 have no I&O index; from 2020-21
    the I&O index is always included (no hold-harmless in the files)."""
    t = {k: np.rint(p[k].to_numpy(float) * 10) for k in ["k8_ai", "k8_pi", "dcai", "k8_io"]}
    dc = p.dcai_present.to_numpy() > 0
    io = (p.k8_io_present.to_numpy() > 0) & (p.year.to_numpy() >= 2021)
    out = np.full(len(p), np.nan)
    for has_dc in (False, True):
        for has_io in (False, True):
            w = K8_WEIGHTS["io" if has_io else "old"][has_dc]
            parts = [t["k8_ai"] * w[0], t["k8_pi"] * w[1],
                     (t["dcai"] if has_dc else 0) * w[2], (t["k8_io"] if has_io else 0) * w[3]]
            m = (dc == has_dc) & (io == has_io)
            out[m] = _round1(sum(np.asarray(x, float) for x in parts))[m]
    ok = ((p.school_type == 1) & p.k8_ai.notna() & p.k8_pi.notna()
          & (~p.dcai_present.astype(bool) | p.dcai.notna())
          & (~(p.k8_io_present.astype(bool) & (p.year >= 2021)) | p.k8_io.notna()))
    return pd.Series(np.where(ok, out, np.nan), index=p.index)


# High-school weights (assessment, progress, ACT/WorkKeys, cohort graduation rate index,
# strength of diploma, I&O) in thousandths; LDOE 2021-22 and 2022-23 SPS calculators.
HS_WEIGHTS = {"old": (125, 125, 250, 250, 250, 0), "io": (125, 125, 250, 200, 250, 50)}
_HS = ["hs_ai", "hs_pi", "act", "cgr_index", "sod", "hs_io"]


def hs_formula(p: pd.DataFrame) -> pd.Series:
    """SPS of a high school with all six indices published (five before 2020-21)."""
    t = [np.rint(p[k].to_numpy(float) * 10) for k in _HS]
    io = p.year.to_numpy() >= 2021
    old = _round1(sum(x * w for x, w in zip(t[:5], HS_WEIGHTS["old"][:5])))
    new = _round1(sum(x * w for x, w in zip(t, HS_WEIGHTS["io"])))
    out = np.where(io, new, old)
    return pd.Series(np.where(p.school_type == 2, out, np.nan), index=p.index)


def build() -> pd.DataFrame:
    """Every school-year on the regular and closed-school sheets, 2018-2025, with the
    computed grade, the K-8 formula SPS, October enrollment and next year's outcomes."""
    if not CACHE.exists():
        df = _build()
        PROCESSED.mkdir(parents=True, exist_ok=True)
        # Write then rename, so a session reading the shared checkout never sees a
        # half-written cache.
        part = CACHE.with_name(f"{CACHE.name}.{id(df)}.part")
        df.to_csv(part, index=False, compression="gzip")
        part.replace(CACHE)
    p = pd.read_csv(CACHE, keep_default_na=False, na_values=[""], dtype={"code": str},
                    low_memory=False)
    for c in ["name", "system", "grade", "grade_computed", "aus", "y_grade"]:
        p[c] = p[c].fillna("").astype(str)
    return p


def _build() -> pd.DataFrame:
    p = pd.concat([_read_year(y) for y in sorted(SPS)], ignore_index=True)
    p = p[p.code.ne("") & ~p.duplicated(["year", "code"], keep="first")]
    cut = np.array([CUTOFFS.get(y, (np.nan,) * 4) for y in p.year], dtype=float)
    for j, g in enumerate("ABCD"):
        p[f"cut_{g}"] = cut[:, j]
    s = p.sps.to_numpy()
    p["grade_computed"] = np.select([s >= cut[:, k] for k in range(4)], list("ABCD"), "F")
    p.loc[p.sps.isna() | ~p.year.isin(GRADED_YEARS), "grade_computed"] = ""
    p["sps_k8_formula"] = k8_formula(p)
    p["sps_hs_formula"] = hs_formula(p)
    mem = pd.concat([_enrollment(y) for y in sorted(ENROLLMENT)], ignore_index=True)
    p = p.merge(mem, on=["year", "code"], how="left", indicator=True)
    p["enrollment_matched"] = (p._merge == "both").astype(float)
    p = p.drop(columns="_merge")
    p["y_year"] = p.year.map(NEXT)
    nxt = p[["code", "year", "sps", "grade", "ai"]].rename(
        columns={"year": "y_year", "sps": "y_sps", "grade": "y_grade", "ai": "y_ai"})
    p = p.merge(nxt, on=["code", "y_year"], how="left", indicator=True)
    p["y_closed"] = np.where(p.y_year.notna() & (p.year < 2025),
                             (p._merge == "left_only") | p.y_sps.isna(), np.nan).astype(float)
    p["y_grade_points"] = p.y_grade.map(GRADE_POINTS)
    p = p.drop(columns="_merge")
    lag = p[["code", "year", "sps"]].rename(columns={"sps": "q_lag"})
    lag["year"] = lag.year.map(NEXT)
    return p.merge(lag.dropna(subset=["year"]).astype({"year": int}), on=["code", "year"],
                   how="left")


# --------------------------------------------------------------------------- checks

def rule_check() -> pd.DataFrame:
    """Published grade against the grade computed from the published SPS and CUTOFFS, by
    year, for every school-year with a published A-F grade on the regular and closed
    sheets; and, for K-8 and high schools with every index unmasked, the published SPS
    against the SPS recomputed from the indices and K8_WEIGHTS / HS_WEIGHTS."""
    p = build()
    rows = []
    for y, g in p.groupby("year"):
        m = g.grade.isin(list("ABCDF"))
        k = g.sps_k8_formula.notna() & g.sps.notna()
        h = g.sps_hs_formula.notna() & g.sps.notna()
        mis = g[m & (g.grade != g.grade_computed)]
        rows.append({
            "year": y, "rows": len(g), "graded": int(m.sum()),
            "turnaround_T": int((g.grade == "T").sum()),
            "no_grade": int((g.grade == "").sum()),
            "agree": float((g.grade[m] == g.grade_computed[m]).mean()) if m.any() else None,
            "mismatch_A_published_B": int(((mis.grade == "B")
                                           & (mis.grade_computed == "A")).sum()),
            "mismatch_other": int(((mis.grade != "B") | (mis.grade_computed != "A")).sum()),
            "k8_unmasked": int(k.sum()),
            "k8_sps_reproduced": float((np.abs(g.sps - g.sps_k8_formula)[k] < 1e-6).mean()),
            "hs_unmasked": int(h.sum()),
            "hs_sps_reproduced": (float((np.abs(g.sps - g.sps_hs_formula)[h] < 1e-6).mean())
                                  if h.any() else None),
        })
    return pd.DataFrame(rows)


# -------------------------------------------------------------------------- samples

def _design(df: pd.DataFrame) -> tuple[np.ndarray, list]:
    X = pd.DataFrame(index=df.index)
    for c in ["pct_ed", "pct_black", "pct_hispanic", "pct_asian", "pct_female", "pct_lep",
              "share_prek_2", "share_6_8", "share_9_12"]:
        X[c] = df[c].fillna(df[c].median())
    X["covariates_missing"] = df.enrollment.isna().astype(float)
    n = df.enrollment.where(df.enrollment > 0)
    X["log_enrollment"] = np.log(n).fillna(np.log(n).median())
    X["has_grade8"] = df.has_grade8.fillna(0.0)
    X["charter"] = df.charter.fillna(0.0)
    X["selective"] = df.selective
    for t in (2, 3):
        X[TYPES[t]] = (df.school_type == t).astype(float)
    for y in sorted(df.year.unique())[1:]:
        X[f"year_{y}"] = (df.year == y).astype(float)
    # Parish of the site (enrollment file), else the school-system prefix of the site
    # code for the 64 parishes; 0 = statewide or unknown.
    code_parish = pd.to_numeric(df.code.str[:3], errors="coerce")
    par = df.parish.fillna(code_parish.where(code_parish <= 64)).fillna(0).astype(int)
    for d in sorted(par.unique()):
        if d != 36:                                        # Orleans is the base
            X[f"parish_{d:02d}"] = (par == d).astype(float)
    X = X.loc[:, X.std() > 0]
    return X.to_numpy(float), list(X.columns)


def _frame(y_col: str, years=MAIN_YEARS) -> pd.DataFrame:
    p = build()
    paired = p.tested_grades.notna() & (p.tested_grades == 0)    # K-2 feeder copies
    df = p[p.year.isin(years) & p.grade.isin(list("ABCDF")) & p.sps.notna() & ~paired
           & (p.alternative == 0) & p[y_col].notna()]
    return df.reset_index(drop=True)


def _sample(boundary: str = MAIN_BOUNDARY, y_col: str = "y_sps") -> RDDSample:
    j = BOUNDARIES[boundary]
    df = _frame(y_col)
    hi, lo = "ABCD"[j], "BCDF"[j]
    c = df[f"cut_{hi}"].to_numpy(float) - HALF
    X, names = _design(df)
    group = df.year.astype(str).to_numpy()
    group_cutoffs = {str(y): CUTOFFS[y][j] - HALF for y in sorted(df.year.unique())}
    return RDDSample(
        Q=df.sps.to_numpy(float) - c, X=X, Y=df[y_col].to_numpy(float), threshold=0.0,
        name=f"la_school_grades_{boundary}_{y_col}", feature_names=names,
        description=(f"Louisiana school performance scores, spring years "
                     f"{', '.join(map(str, MAIN_YEARS))}. Q = published SPS minus "
                     f"(own {hi}-cutoff - 0.05); treatment = 1{{Q >= 0}} (grade {hi} or "
                     f"better rather than {lo}); Y = {y_col} (next SPS year)."),
        citation="Louisiana Department of Education, School Performance Scores 2018-2025 "
                 "and Multiple Statistics by Site (October 1 enrollment)",
        treatment_rule=lambda Q: (Q >= 0).astype(int),
        extras={
            "group": group, "group_cutoffs": group_cutoffs,
            "boundary": boundary, "cutoff_table": CUTOFFS,
            "year": df.year.to_numpy(int), "school_type": df.school_type.to_numpy(int),
            "code": df.code.to_numpy(str), "q_raw": df.sps.to_numpy(float),
            "grade_published": df.grade.to_numpy(str),
            "cutoffs": df[["cut_A", "cut_B", "cut_C", "cut_D"]].to_numpy(float) - HALF,
            "eligible_published": df.grade.isin(list("ABCD"[:j + 1])).to_numpy(float),
            "y_year": df.y_year.to_numpy(int), "q_lag": df.q_lag.to_numpy(float),
            "closed_sheet": df.closed_sheet.to_numpy(float),
            "outcome_simulated": (df.y_year == 2021).to_numpy(float),
        },
    )


def load() -> RDDSample:
    """B/C boundary; Y = next SPS year's published SPS."""
    return _sample(MAIN_BOUNDARY)


def load_ab() -> RDDSample:
    return _sample("AB")


def load_bc() -> RDDSample:
    return _sample("BC")


def load_cd() -> RDDSample:
    return _sample("CD")


def load_df() -> RDDSample:
    return _sample("DF")


def load_next_assessment() -> RDDSample:
    """Y = next year's K-8 & high-school assessment index."""
    return _sample(MAIN_BOUNDARY, "y_ai")


def load_next_grade() -> RDDSample:
    """Y = next grade (A=4 ... F=0). Missing for 2018-19, since 2020-21 has no grades."""
    return _sample(MAIN_BOUNDARY, "y_grade_points")


def load_closed() -> RDDSample:
    """Y = 1 if the school has no SPS in the next SPS year's file."""
    return _sample(MAIN_BOUNDARY, "y_closed")


def load_two_score(boundary: str = MAIN_BOUNDARY) -> RDDSample:
    """Assessment against progress, for K-8 schools with no grade-8 (DCAI) component in
    2017-18 and 2018-19, before the I&O index. Their SPS is round_0.1(0.75 Q1 + 0.25 Q2),
    Q1 = K-8 assessment index and Q2 = K-8 progress index (both published to 0.1), so the
    grade is B or better exactly when 0.75 Q1 + 0.25 Q2 >= 74.95."""
    j = BOUNDARIES[boundary]
    df = _frame("y_sps", years=(2018, 2019))
    df = df[(df.school_type == 1) & (df.dcai_present == 0) & df.k8_ai.notna()
            & df.k8_pi.notna()].reset_index(drop=True)
    q = df[["k8_ai", "k8_pi"]].to_numpy(float)
    cut = CUTOFFS[2018][j] - HALF
    w = (0.75, 0.25)
    X, names = _design(df)
    return RDDSample(
        Q=q, X=X, Y=df.y_sps.to_numpy(float), threshold=(cut, cut),
        name=f"la_school_grades_two_score_{boundary}", feature_names=names,
        description=("Louisiana K-8 schools without a grade-8 component, 2018 and 2019. "
                     "Q = (assessment index, progress index); treatment = "
                     f"1{{0.75 Q1 + 0.25 Q2 >= {cut}}}; Y = next SPS."),
        citation="Louisiana Department of Education, School Performance Scores",
        # 1e-9 absorbs floating-point error in sums that sit exactly on the cutoff.
        treatment_rule=lambda Q: (Q @ np.array(w) >= cut - 1e-9).astype(int),
        extras={"weights": w, "cutoff": cut, "year": df.year.to_numpy(int),
                "sps_published": df.sps.to_numpy(float),
                "eligible_published": df.grade.isin(list("ABCD"[:j + 1])).to_numpy(float)},
    )
