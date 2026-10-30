"""Readers for FLDOE's 2001-2014 school-grades spreadsheets.

Each year's file has its own column names, so ``read_year`` maps them onto one set of
canonical columns. Components are whole-number percentages (points). Missing or "NA"
cells become NaN. The 2010-2014 releases come in two files (non-high schools and high
schools); both are read and stacked, with ``file`` = "nonhigh" or "high".

Canonical columns (not every year has every column):

* identifiers: year, district, school, name, school_type (1 elementary, 2 middle,
  3 high, 4 combination), file;
* components: read_prof, math_prof, write_prof, sci_prof (percent meeting high
  standards), read_gain, math_gain (percent making learning gains), read_l25,
  math_l25 (learning gains of the lowest 25%), bonus (high-school retake bonus, 0 or 10);
* totals: points_pub (assessment points as published), points_final (the total the
  grade is read from, after any weighting or adjustment), scale (maximum points);
* conditions: pct_tested, ap_read_cur, ap_read_prev, ap_read_impr, ap_math_cur,
  ap_math_prev, ap_math_impr (adequate progress of the lowest 25%: Y/N/""),
  read25_met (2013-2014 reading performance requirement), protected (one-letter-grade
  drop protection used), ms_accel (middle-school acceleration counted), hs_points;
* published grade and grade history: grade, grade_prev;
* school characteristics printed in the file: frl, minority (percent), charter, title1,
  alt_ese, region.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.datasets.fl_school_grades._spreadsheet import read
from experiments.datasets.fl_grades_2002.download import RAW, SPREADSHEETS

TYPE_CODES = {"ELEM": 1, "MIDL": 2, "HIGH": 3, "COMB": 4, "E": 1, "M": 2, "H": 3, "C": 4}

# Exact header -> canonical name (headers are whitespace-normalised first).
_EXACT = {
    "District Number": "district", "School Number": "school", "School Name": "name",
    "SCHNAME": "name", "schlname": "name", "School Type": "school_type", "Type": "school_type",
    "Percent Tested": "pct_tested", "PCTST03": "pct_tested", "pctst04": "pct_tested",
    "Charter": "charter", "Title I": "title1", "Region": "region",
    "Free or Reduced Lunch Rate": "frl", "FRPL": "frl", "Free or Reduced-Price Lunch": "frl",
    "FRL": "frl", "Minority Rate": "minority", "Minority%": "minority", "MinRate": "minority",
    # 2003 file (SAS-style names)
    "READPRF": "read_prof", "MATHPRF": "math_prof", "WRITEPRF": "write_prof",
    "READGN": "read_gain", "MATHGN": "math_gain", "LOPTS": "read_l25", "GRPTS": "points_pub",
    # 2004 file
    "Rprf": "read_prof", "Mprf": "math_prof", "Wprf": "write_prof", "Rgn": "read_gain",
    "Mgn": "math_gain", "lowpoints": "read_l25", "pts04": "points_pub", "adq04": "ap_read_cur",
    # proficiency
    "% at Level 3 or Higher in Reading": "read_prof",
    "% Meeting High Standards in Reading": "read_prof",
    "Reading % Satisfactory or Higher": "read_prof",
    "% at Level 3 or Higher in Math": "math_prof",
    "% Meeting High Standards in Math": "math_prof",
    "Math % Satisfactory or Higher": "math_prof",
    "% Meeting the Writing Standard": "write_prof",
    "% Meeting High Standards in Writing": "write_prof",
    "Writing % Satisfactory or Higher": "write_prof",
    "% Meeting High Standards in Science": "sci_prof",
    "% at Level 3 or Higher in Science": "sci_prof",
    "Science % Satisfactory or Higher": "sci_prof",
    # gains
    "% Making Learning Gains in Reading": "read_gain", "% Making Reading Gains": "read_gain",
    "Reading Points for Gains": "read_gain",
    "% Making Learning Gains in Math": "math_gain", "% Making Math Gains": "math_gain",
    "Math Points for Gains": "math_gain",
    "% of Lowest 25% Making Learning Gains in Reading": "read_l25",
    "Reading Gains for Low 25%": "read_l25",
    "% of Lowest 25% Making Learning Gains in Math": "math_l25",
    "Math Gains for Low 25%": "math_l25",
    "11th and 12th Grade FCAT Retake Bonus Points": "bonus",
    "High School Retake Bonus Points": "bonus",
    # totals
    "Points Earned (Sum of Previous 6 Columns)": "points_pub",
    "Points Earned (Sum of Previous 9 Columns)": "points_pub",
    "FCAT Points Earned (Sum of Previous 9 Columns)": "points_pub",
    "Points Earned": "points_pub",
    "Sum Points Earned": "points_sum",                          # 2012
    "Applied Points": "points_applied",                         # 2012
    "Applied Points Plus Bonus Points": "points_pub",           # 2012
    "Adjusted Points Earned": "points_adjusted",                # 2012 non-high
    "Final Adjusted Points": "points_adjusted",                 # 2012 high
    "Total Points, Including Adjusted Points if Applicable": "points_adjusted",
    "Total Points, Including Adkusted Points If Applicable": "points_adjusted",
    "Total Points Including Adjusted Points, If Applicable": "points_adjusted",
    "Total Points (FCAT Points + HS Points)": "points_total",    # 2010-2011 high
    "Total Applied Points (Assessment Points + HS Points)": "points_total",
    "High School Component Points Subtotal": "hs_points",
    "Weighting FCAT/HS Points": "weighting", "Weighting Assessment/HS Points": "weighting",
    "Scale": "scale",
    "Middle School Acceleration Qualification": "ms_accel",
    "Middle School Acceleration Participation": "ms_accel",
    "Middle School Acceleration Participation Points": "ms_part_points",
    "Middle School Acceleration Performance Points": "ms_perf_points",
    "Did this School Benefit from the One-Letter-Grade-Drop Protection?": "protected",
    "Did this school benefit from the one-letter-grade-drop protection?": "protected",
    "25% or More of FCAT 2.0 Readers Scoring Satisfactory or Higher": "read25_met",
    "25% or More of FCAT 2x0 Readers Scoring Satisfactory or Higher": "read25_met",
    "FCAT Reading % Satisfactory or Higher": "read_prof_fcat",
    "FCAT 2.0 Reading % Satisfactory or Higher": "read_prof_fcat",
    "Adequate Progress for At-Risk? (Yes/No)": "ap_at_risk",
    "Improvement of the Lowest 25% in Reading": "ap_read_impr",
    "Improvement of the Lowest 25% in Math": "ap_math_impr",
    "50% or More of Lowest 25% of Readers Making Learning Gains": "ap_read_cur",
}
_ALT_ESE = re.compile(r"^A ?- ?Alternative School")


def _norm(h) -> str:
    return " ".join(str(h).split()) if h is not None else ""


def _canon(h: str, year: int) -> str | None:
    if h in _EXACT:
        return _EXACT[h]
    if _ALT_ESE.match(h):
        return "alt_ese"
    m = re.match(r"^(?:Grade|GR|Gr) ?(\d{2,4})$", h)
    if m:
        y = int(m.group(1))
        y = y + 2000 if y < 50 else (y + 1900 if y < 100 else y)
        return {year: "grade", year - 1: "grade_prev", year - 2: "grade_prev2"}.get(y)
    # Adequate progress of the lowest 25%, with the year in the header.
    m = re.match(r"^(?:50% or More of Lowest 25% of (Readers|Students) Making Learning Gains"
                 r"(?: in Math)? - |(Reading|Math) Gains for Lowest 25%, at Least 50 Points "
                 r"in )(\d{4})$", h)
    if m:
        subj = "read" if (m.group(1) == "Readers" or m.group(2) == "Reading") else "math"
        when = "cur" if int(m.group(3)) == year else "prev"
        return f"ap_{subj}_{when}"
    return None


def _frame(path: Path) -> pd.DataFrame:
    rows = read(path, 0)
    h = next((k for k, r in enumerate(rows[:12])
              if r and any(_norm(x) in ("District Number", "DistrictName", "DISTSCHL")
                           for x in r)), None)
    if h is None:
        raise ValueError(f"no header row in {path}")
    hdr = [_norm(x) or f"col{j}" for j, x in enumerate(rows[h])]
    body = [r[:len(hdr)] + [None] * (len(hdr) - len(r)) for r in rows[h + 1:]]
    df = pd.DataFrame(body, columns=hdr)
    return df.dropna(how="all")


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.strip().replace({"": None, "None": None, "NA": None,
                                                            "N/A": None, "nan": None}),
                         errors="coerce")


def _txt(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().replace({"None": "", "nan": ""})


def _yn(s: pd.Series) -> pd.Series:
    v = _txt(s).str.upper()
    return v.map({"YES": "Y", "Y": "Y", "NO": "N", "N": "N"}).fillna("")


def read_file(path: Path, year: int, kind: str) -> pd.DataFrame:
    raw = _frame(path)
    cols: dict[str, str] = {}
    for c in raw.columns:
        k = _canon(c, year)
        if k is not None and k not in cols.values():
            cols[c] = k
    df = raw[list(cols)].rename(columns=cols)
    out = pd.DataFrame(index=raw.index)
    out["year"] = year
    out["file"] = kind
    if "DistrictName" in raw:                      # 2003: COSCH = district * 10000 + school
        code = _num(raw.COSCH)
        out["district"], out["school"] = code // 10000, code % 10000
    elif "DISTSCHL" in raw:                        # 2004: "DDSSSS"
        code = _txt(raw.DISTSCHL).str.zfill(6)
        out["district"], out["school"] = _num(code.str[:2]), _num(code.str[2:])
    else:
        out["district"], out["school"] = _num(df.district), _num(df.school)
    out = out[out.district.notna() & out.school.notna() & (out.district > 0)].copy()
    df = df.loc[out.index]
    out["district"] = out.district.astype(int)
    out["school"] = out.school.astype(int)
    out["name"] = _txt(df["name"])
    st = _txt(df["school_type"]).str.upper()
    out["school_type"] = st.map(TYPE_CODES).fillna(_num(st)).astype(float)
    for c in ["read_prof", "math_prof", "write_prof", "sci_prof", "read_gain", "math_gain",
              "read_l25", "math_l25", "bonus", "points_pub", "points_sum", "points_applied",
              "points_adjusted", "points_total", "hs_points", "scale", "ms_part_points",
              "ms_perf_points", "read_prof_fcat", "pct_tested", "frl", "minority", "region"]:
        out[c] = _num(df[c]) if c in df else np.nan
    for c in ["grade", "grade_prev", "grade_prev2", "weighting"]:
        out[c] = _txt(df[c]).str.upper() if c in df else ""
    for c in ["ap_read_cur", "ap_read_prev", "ap_read_impr", "ap_math_cur", "ap_math_prev",
              "ap_math_impr", "read25_met", "protected", "ms_accel", "ap_at_risk", "charter",
              "title1"]:
        out[c] = _yn(df[c]) if c in df else ""
    out["alt_ese"] = _txt(df["alt_ese"]).str.upper() if "alt_ese" in df else ""
    # 2003 publishes the minority and FRL rates as fractions.
    if year == 2003:
        out["frl"] *= 100
        out["minority"] *= 100
    return out.reset_index(drop=True)


def read_year(year: int) -> pd.DataFrame:
    names = SPREADSHEETS[year]
    frames = []
    for n in names:
        path = RAW / n
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing. Run python -m experiments.datasets.fl_grades_2002.download")
        kind = "high" if n.startswith("HighSchool") else ("nonhigh" if n.startswith("NonHigh")
                                                         else "all")
        frames.append(read_file(path, year, kind))
    return pd.concat(frames, ignore_index=True)
