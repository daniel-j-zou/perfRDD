"""California School Dashboard (CDE): a 5 x 5 decision matrix of status and change, for the
ELA and mathematics academic indicators.

Each school gets a status on each academic indicator: the average Distance from Standard
(DFS; "Distance from Proficiency" from 2026), the mean scale-score distance of its
continuously enrolled test takers (grades 3-8 and 11) from the lowest Level 3 ("Standard
Met") score, rounded to a tenth of a point. Change is the status minus the prior year's
status, both as CDE computes them in the current year (the prior is recomputed under
the current year's rules). Status and change are rounded to a tenth separately, so the
published change can differ by 0.1 from the difference of the published statuses. The Dashboard Technical Guide ("Status Cut Scores, Change
Cut Scores, and Five-by-Five Colored Tables") turns status into five status levels and
change into five change levels, and a 5 x 5 table turns the pair into a color:

    status level   Declined Sig.  Declined  Maintained  Increased  Increased Sig.
    Very High      Green          Green     Blue        Blue       Blue
    High           Green          Green     Green       Green      Blue
    Medium         Yellow         Yellow    Yellow      Green      Green
    Low            Orange         Orange    Orange      Yellow     Yellow
    Very Low       Red            Red       Red         Orange     Orange

Change levels (all years, both cut-score sets): Declined Significantly <= -15.1,
Declined -15.0 to -3.0, Maintained -2.9 to +2.9, Increased +3.0 to +14.9, Increased
Significantly >= +15.0. Status levels: lowest Low / Medium / High / Very High status

    ELA,  grades 3-8 cut scores:  -70.0 / -5.0  / +10.0 / +45.0
    ELA,  grade 11 (high school): -45.0 /  0.0  / +30.0 / +75.0
    math, grades 3-8 cut scores:  -95.0 / -25.0 /  0.0  / +35.0
    math, grade 11 (high school): -115.0 / -60.0 /  0.0 / +25.0

The table, the change cuts and the status cuts are the same in every Dashboard year
2017-2026 (2023, 2024 and 2026 guides; the published levels in the 2017-2025 files).
High schools and high-school districts use the grade 11 cut scores; elementary, middle
and K-12 schools use the grades 3-8 ones (``hscutpoints`` in the files). A school gets a
color only with 30 or more students in the current and the prior year's denominator.

D = Red, a Very Low status with a change below +3.0 (the three Red cells). In status
    and change this is a quadrant ("both" rule), not a staircase. Red feeds
    Differentiated Assistance for districts and CSI / ATSI identification for schools,
    and it is the color published on the Dashboard. ``level="red_orange"`` gives Red or
    Orange, a staircase: Very Low with any change, or Low with a change below +3.0.
Q = ``load_two_score()``: (status relative to the Very Low / Low boundary, change).
    Status is centred at the lowest Low status of the school's cut-score set (-70.0 for
    ELA grades 3-8, -45.0 for ELA grade 11, ...), so that Red is
    1{Q1 < -0.05} * 1{Q2 < 2.95} for every school; both are on a 0.1 grid, so the
    0.05 margins are exact. ``load()``: Q = the same centred status, with the prior
    status (prior = status - change) in X and in extras; D = 1{Q < min(-0.05,
    prior + 2.95)}, a cutoff on status that shifts with the prior status.
X = school characteristics not computed from that year's tests, from the Dashboard
    census-enrollment file of the same school year (Census Day, October): % socio-
    economically disadvantaged, % English learners, % students with disabilities,
    % foster youth, % homeless, race and ethnicity shares, log enrollment; the
    district's log enrollment and % socioeconomically disadvantaged; charter,
    high-school cut scores, census-missing flags; county and year indicators (the
    county-office flag is never set in the sample, so it drops out).
    ``load()`` adds the prior status; ``load_two_score()`` leaves it out (with it, the
    change would be status minus a covariate, and the two noises would coincide).
Y = the next Dashboard year's status on the same indicator, matched by the 14-digit CDS
    code (2019 -> 2022: no Dashboard in 2020 and 2021).

Sample: schools (all students) with a published color (1-5) in the 2017 (Fall), 2018,
2019, 2023, 2024 and 2025 Dashboards, excluding Dashboard Alternative School Status
(DASS) schools, pair-and-share schools (K-2 schools assigned a feeder's grade 3 status)
and the 2018-2019 "no test" schools (Orange without status or change), with Y observed.
2022 reported status only, with no change and no colors: it is used only as the outcome
for 2019 and as the prior inside the 2023 file (``load_status_2022()`` gives the 2022
Very Low status as a single-cutoff sample). The 2017 grade 11 files carry no levels or
colors, so 2017 has grades 3-8 schools only. 2026 is used only as the outcome for 2025.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample
from experiments.datasets.ca_dashboard.download import RAW, census_file, indicator_files

HERE = Path(__file__).parent
PROCESSED = HERE / "data" / "processed"
CACHE = PROCESSED / "ca_dashboard_panel.csv.gz"

SUBJECTS = ("ela", "math")
YEARS = (2017, 2018, 2019, 2022, 2023, 2024, 2025, 2026)      # Dashboard years
NEXT = dict(zip(YEARS[:-1], YEARS[1:]))
COLOR_YEARS = (2017, 2018, 2019, 2023, 2024, 2025, 2026)       # colors published
MAIN_YEARS = (2017, 2018, 2019, 2023, 2024, 2025)               # colors and an outcome

COLORS = {1: "Red", 2: "Orange", 3: "Yellow", 4: "Green", 5: "Blue"}
STATUS_LEVELS = ("Very Low", "Low", "Medium", "High", "Very High")
CHANGE_LEVELS = ("Declined Significantly", "Declined", "Maintained", "Increased",
                 "Increased Significantly")
# MATRIX[s - 1][c - 1] = color code for status level s and change level c (1 = Red ...
# 5 = Blue). The same table in every year 2017-2026.
MATRIX = (
    (1, 1, 1, 2, 2),    # Very Low
    (2, 2, 2, 3, 3),    # Low
    (3, 3, 3, 4, 4),    # Medium
    (4, 4, 4, 4, 5),    # High
    (4, 4, 5, 5, 5),    # Very High
)
TREATED_COLORS = {"red": (1,), "red_orange": (1, 2)}
# Lowest status (DFS points) of the Low, Medium, High and Very High levels.
STATUS_MIN = {
    ("ela", "k8"): (-70.0, -5.0, 10.0, 45.0),
    ("ela", "hs"): (-45.0, 0.0, 30.0, 75.0),
    ("math", "k8"): (-95.0, -25.0, 0.0, 35.0),
    ("math", "hs"): (-115.0, -60.0, 0.0, 25.0),
}
# Lowest change of the Declined, Maintained, Increased and Increased Significantly levels.
CHANGE_MIN = (-15.0, -2.9, 3.0, 15.0)
GRID = 0.05            # half the 0.1 reporting grid: boundaries sit between grid points
N_MIN = 30             # students in the current and the prior denominator for a color

CENSUS_GROUPS = {"SED": "pct_sed", "EL": "pct_el", "SWD": "pct_swd", "FOS": "pct_foster",
                 "HOM": "pct_homeless", "AA": "pct_black", "AI": "pct_amind",
                 "AS": "pct_asian", "FI": "pct_filipino", "HI": "pct_hispanic",
                 "PI": "pct_pacisl", "MR": "pct_multi", "WH": "pct_white"}
X_SHARES = ["pct_sed", "pct_el", "pct_swd", "pct_foster", "pct_homeless", "pct_black",
            "pct_amind", "pct_asian", "pct_filipino", "pct_hispanic", "pct_pacisl",
            "pct_multi"]                                   # white is the omitted share
_STR = ["cds", "subject", "statuslevel", "changelevel", "color", "pairshare",
        "y_pairshare"]


# ------------------------------------------------------------------------ raw files

def _read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run python -m experiments.datasets.ca_dashboard.download")
    d = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                    encoding="utf-8-sig", encoding_errors="replace")
    d.columns = [c.strip().lower() for c in d.columns]
    for c in d.columns:
        d[c] = d[c].str.strip()
    return d


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _flag(d: pd.DataFrame, col: str) -> pd.Series:
    return d[col].eq("Y").astype(float) if col in d else pd.Series(0.0, index=d.index)


def _read_indicator(subject: str, year: int) -> pd.DataFrame:
    """All-students school rows of one indicator file (for 2017, the grades 3-8 file)."""
    d = _read_tsv(RAW / indicator_files(subject, year)[0])
    d = d[(d.rtype == "S") & (d.studentgroup == "ALL")].copy()
    out = pd.DataFrame({"cds": d.cds, "subject": subject, "year": year})
    out["charter"] = _flag(d, "charter_flag")
    out["coe"] = _flag(d, "coe_flag")
    out["dass"] = _flag(d, "dass_flag")
    out["hs"] = _flag(d, "hscutpoints")
    out["notest"] = _flag(d, "notestflag")
    out["pairshare"] = d.get("pairshare_method", pd.Series("", index=d.index))
    for a, b in [("currdenom", "n_curr"), ("priordenom", "n_prior"), ("currstatus", "status"),
                 ("priorstatus", "prior_published"), ("change", "change")]:
        out[b] = _num(d[a]) if a in d else np.nan
    for c in ["statuslevel", "changelevel", "color"]:
        out[c] = d[c].replace("", "0") if c in d else "0"
    # Schools that did not test (``notestflag``, 2019) were published Orange with no
    # change level; 2018 has no flag, but the same rows (a color with change level 0).
    out["notest"] = np.fmax(out.notest, (out.color.isin(list("12345"))
                                         & out.changelevel.eq("0")).astype(float))
    return out.reset_index(drop=True)


def _read_census(year: int) -> pd.DataFrame:
    """School and district enrollment and student-group shares (Census Day)."""
    d = _read_tsv(RAW / census_file(year))
    d = d[d.rtype.isin(["S", "D"])]
    d = d.assign(rate=_num(d.rate), total=_num(d.totalenrollment))
    wide = d.pivot_table(index=["cds", "rtype"], columns="studentgroup", values="rate",
                         aggfunc="first").rename(columns=CENSUS_GROUPS)
    tot = d.groupby(["cds", "rtype"]).total.max()
    wide["enrollment"] = tot
    wide = wide.reset_index()
    wide["year"] = year
    return wide


# ---------------------------------------------------------------------------- rule

def status_level(status_c: pd.Series | np.ndarray, cuts_c) -> np.ndarray:
    """Status level 1-5 from the centred status and the centred lowest Low, Medium, High,
    Very High statuses (all on the 0.1 grid; boundaries at cut - 0.05)."""
    s = np.asarray(status_c, float)
    cuts = np.asarray(cuts_c, float)
    lev = 1 + (s[:, None] > cuts.reshape(-1, 4) - GRID).sum(axis=1)
    return np.where(np.isnan(s), 0, lev)


def change_level(change: pd.Series | np.ndarray) -> np.ndarray:
    c = np.asarray(change, float)
    lev = 1 + (c[:, None] > np.array(CHANGE_MIN)[None, :] - GRID).sum(axis=1)
    return np.where(np.isnan(c), 0, lev)


def matrix_color(slev: np.ndarray, clev: np.ndarray) -> np.ndarray:
    m = np.array(MATRIX)
    ok = (slev > 0) & (clev > 0)
    out = np.zeros(len(slev), int)
    out[ok] = m[slev[ok] - 1, clev[ok] - 1]
    return out


def matrix_rule(level: str = "red", status_cuts: np.ndarray | None = None):
    """D(Q) = 1{the 5 x 5 table gives a treated color} for Q = (centred status, change).

    Red is status level 1 (centred status < -0.05) and change level 1-3 (change < 2.95).
    The Very Low / Low boundary is 0 in centred units for every cut-score set, so the
    Red rule needs no unit-specific input. Red-or-Orange also uses the Low / Medium
    boundary, which differs between the grades 3-8 and grade 11 cut scores, so it takes
    ``status_cuts``: the centred lowest Low, Medium, High and Very High statuses, shape
    (4,) or (n, 4) for the n units the rule is applied to."""
    treated = np.array(TREATED_COLORS[level])
    if level != "red" and status_cuts is None:
        raise ValueError(f"level={level!r} needs status_cuts")

    def rule(Q: np.ndarray) -> np.ndarray:
        Q = np.asarray(Q, float).reshape(-1, 2)
        if level == "red":
            return ((Q[:, 0] < -GRID) & (Q[:, 1] < CHANGE_MIN[2] - GRID)).astype(int)
        slev = status_level(Q[:, 0], status_cuts)
        clev = change_level(Q[:, 1])
        return np.isin(matrix_color(slev, clev), treated).astype(int)
    return rule


def _status_offset(df: pd.DataFrame) -> np.ndarray:
    """The lowest Low status of each row's cut-score set (the centring constant)."""
    grp = np.where(df.hs == 1, "hs", "k8")
    return np.array([STATUS_MIN[(s, g)][0] for s, g in zip(df.subject, grp)], float)


def _status_cuts_c(df: pd.DataFrame) -> np.ndarray:
    grp = np.where(df.hs == 1, "hs", "k8")
    raw = np.array([STATUS_MIN[(s, g)] for s, g in zip(df.subject, grp)], float)
    return np.round(raw - raw[:, :1], 1)


def apply_rule(p: pd.DataFrame) -> pd.DataFrame:
    """Computed status level, change level and color for every row, from the published
    status and change, with the n-size rule (>= 30 students in both denominators)."""
    out = pd.DataFrame(index=p.index)
    off = _status_offset(p)
    out["status_offset"] = off
    out["status_c"] = np.round(p.status - off, 1)
    out["slev_calc"] = status_level(out.status_c, _status_cuts_c(p))
    out["clev_calc"] = change_level(p.change)
    color = matrix_color(out.slev_calc.to_numpy(), out.clev_calc.to_numpy())
    nsize = (p.n_curr >= N_MIN) & (p.n_prior >= N_MIN)
    out["nsize_met"] = nsize.astype(float)
    out["color_calc"] = np.where(nsize, color, 0)
    return out


# --------------------------------------------------------------------------- panel

def build() -> pd.DataFrame:
    """Every school-year-subject (all students) of the Dashboard files 2017-2026 with the
    published and computed levels and colors, the census covariates, and the next
    Dashboard year's status."""
    if not CACHE.exists():
        df = _build()
        PROCESSED.mkdir(parents=True, exist_ok=True)
        # Atomic write: other processes may load the registry concurrently.
        fd, tmp = tempfile.mkstemp(prefix=".ca_dashboard_", suffix=".csv.gz", dir=PROCESSED)
        os.close(fd)
        try:
            df.to_csv(tmp, index=False, compression="gzip")
            os.chmod(tmp, 0o644)                    # mkstemp creates the file 0600
            os.replace(tmp, CACHE)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    p = pd.read_csv(CACHE, keep_default_na=False, na_values=[""], low_memory=False,
                    dtype={c: str for c in _STR})
    for c in _STR:
        p[c] = p[c].fillna("").astype(str)
    return p


def _build() -> pd.DataFrame:
    p = pd.concat([_read_indicator(s, y) for s in SUBJECTS for y in YEARS],
                  ignore_index=True)
    p = pd.concat([p, apply_rule(p)], axis=1)
    p["county"] = p.cds.str[:2]
    p["district"] = p.cds.str[:7]
    p["published_color"] = p.color.isin(list("12345")).astype(float)

    cen = pd.concat([_read_census(y) for y in YEARS], ignore_index=True)
    sch = cen[cen.rtype == "S"].drop(columns="rtype")
    dist = cen[cen.rtype == "D"][["cds", "year", "enrollment", "pct_sed"]].rename(
        columns={"enrollment": "district_enrollment", "pct_sed": "district_pct_sed"})
    dist["district"] = dist.cds.str[:7]
    p = p.merge(sch, on=["cds", "year"], how="left")
    p = p.merge(dist.drop(columns="cds"), on=["district", "year"], how="left")

    # Next Dashboard year's status, same school and subject.
    p["y_year"] = p.year.map(NEXT)
    nxt = p[["cds", "subject", "year", "status", "statuslevel", "color", "pairshare",
             "n_curr"]].rename(columns={
        "year": "y_year", "status": "y_status", "statuslevel": "y_statuslevel",
        "color": "y_color", "pairshare": "y_pairshare", "n_curr": "y_n"})
    p = p.merge(nxt, on=["cds", "subject", "y_year"], how="left", indicator=True)
    p["y_missing_row"] = np.where(p.y_year.notna(), (p._merge == "left_only").astype(float),
                                  np.nan)
    p = p.drop(columns="_merge")
    # A pair-and-share school's status is its feeder's, not its own outcome.
    p.loc[p.y_pairshare.fillna("") != "", "y_status"] = np.nan
    return p.sort_values(["subject", "year", "cds"]).reset_index(drop=True)


# -------------------------------------------------------------------------- checks

def _int(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int).to_numpy()


def rule_check() -> pd.DataFrame:
    """Per subject and Dashboard year, all-students school rows: the published status
    level, change level and color against the rule computed from the published status,
    change and denominators. ``dass2019`` rows (DASS schools in 2019, which had their
    own Very Low cut scores) are left out of the agreement rates and reported apart."""
    p = build()
    rows = []
    for (subj, y), g in p.groupby(["subject", "year"]):
        pub_s = g.statuslevel.isin(list("12345"))
        pub_c = g.changelevel.isin(list("12345"))
        pub_col = g.color.isin(list("12345"))
        d19 = (g.dass == 1) & (y == 2019)
        regular = ~d19 & (g.pairshare == "") & (g.notest == 0)
        slev = g.slev_calc.astype(str)
        clev = g.clev_calc.astype(str)
        col_from_levels = pd.Series(matrix_color(_int(g.statuslevel), _int(g.changelevel)),
                                    index=g.index).astype(str)
        has_color = y in COLOR_YEARS
        rows.append({
            "subject": subj, "year": y, "schools": len(g),
            "published_color": int(pub_col.sum()) if has_color else 0,
            "red": int((g.color == "1").sum()), "orange": int((g.color == "2").sum()),
            "dass": int((g.dass == 1).sum()), "pairshare": int((g.pairshare != "").sum()),
            "notest_orange": int(((g.notest == 1) & pub_col).sum()),
            "status_level_agree": float((slev == g.statuslevel)[pub_s & ~d19].mean()),
            "status_level_rows": int((pub_s & ~d19).sum()),
            "dass2019_status_agree": float((slev == g.statuslevel)[pub_s & d19].mean())
            if d19.any() else np.nan,
            "change_level_agree": float((clev == g.changelevel)[pub_c].mean())
            if has_color else np.nan,
            "matrix_agree": float((col_from_levels == g.color)[pub_col & regular].mean())
            if has_color else np.nan,
            "color_agree": float((g.color_calc.astype(str) == g.color)[regular].mean())
            if has_color else np.nan,
            "color_rows": int(regular.sum()) if has_color else 0,
            "red_agree": float(((g.color_calc == 1) == (g.color == "1"))[regular].mean())
            if has_color else np.nan,
        })
    return pd.DataFrame(rows)


def student_group_rows(subject: str = "ela", years=MAIN_YEARS) -> pd.DataFrame:
    """School-level student-group rows (not all students) with a published color, per
    year: the size of a student-group extension."""
    out = []
    for y in years:
        d = _read_tsv(RAW / indicator_files(subject, y)[0])
        g = d[(d.rtype == "S") & (d.studentgroup != "ALL") & d.color.isin(list("12345"))]
        out.append({"year": y, "rows": len(g), "red": int((g.color == "1").sum()),
                    "groups": g.studentgroup.nunique()})
    return pd.DataFrame(out)


# ------------------------------------------------------------------------- samples

def _design(df: pd.DataFrame, prior: bool = False) -> tuple[np.ndarray, list]:
    X = pd.DataFrame(index=df.index)
    school = X_SHARES + ["log_enrollment"]
    district = ["district_log_enrollment", "district_pct_sed"]
    v = df[X_SHARES].copy()
    v["log_enrollment"] = np.log(df.enrollment.where(df.enrollment > 0))
    v["district_log_enrollment"] = np.log(df.district_enrollment.where(df.district_enrollment > 0))
    v["district_pct_sed"] = df.district_pct_sed
    for c in school + district:
        X[c] = v[c].fillna(v[c].median())
    # One flag per source: a school or district missing from the census file.
    X["census_missing"] = v[school].isna().any(axis=1).astype(float)
    X["district_census_missing"] = v[district].isna().any(axis=1).astype(float)
    if prior:
        X["prior_status"] = df.prior_c
    X["charter"] = df.charter
    X["coe_school"] = df.coe
    X["hs_cut_scores"] = df.hs
    for y in sorted(df.year.unique())[1:]:
        X[f"year_{y}"] = (df.year == y).astype(float)
    for c in sorted(df.county.unique()):
        if c != "19":                                  # Los Angeles is the base
            X[f"county_{c}"] = (df.county == c).astype(float)
    X = X.loc[:, X.std() > 0]
    return X.to_numpy(float), list(X.columns)


def _frame(subject: str, years=MAIN_YEARS, cut_set: str = "all") -> pd.DataFrame:
    p = build()
    m = (p.subject.eq(subject) & p.year.isin(years) & p.color.isin(list("12345"))
         & (p.dass == 0) & (p.pairshare == "") & (p.notest == 0) & p.y_status.notna())
    if cut_set == "k8":
        m &= p.hs == 0
    elif cut_set == "hs":
        m &= p.hs == 1
    elif cut_set != "all":
        raise ValueError(f"cut_set must be 'all', 'k8' or 'hs', not {cut_set!r}")
    df = p[m].reset_index(drop=True)
    # Prior status implied by the published change, centred like the status. It equals
    # the published prior status (centred) up to the 0.1 rounding of the change.
    df["prior_c"] = np.round(df.status_c - df.change, 1)
    return df


def _matrix_extras(subject: str, df: pd.DataFrame) -> dict:
    grp = np.where(df.hs == 1, "hs", "k8")
    raw = {g: STATUS_MIN[(subject, g)] for g in ("k8", "hs")}
    return {
        "rule": "matrix",
        "matrix_colors": [[COLORS[c] for c in row] for row in MATRIX],
        "matrix_codes": [list(r) for r in MATRIX],
        "status_levels": STATUS_LEVELS, "change_levels": CHANGE_LEVELS,
        "red_cells": [(1, 1), (1, 2), (1, 3)],      # (status level, change level), 1-based
        "orange_cells": [(1, 4), (1, 5), (2, 1), (2, 2), (2, 3)],
        # Lowest value of status levels 2-5 (Low, Medium, High, Very High) and change
        # levels 2-5, on the 0.1 grid; the level boundaries are these minus 0.05.
        "status_level_min_raw": raw,
        "status_level_min": {g: tuple(np.round(np.array(v) - v[0], 1)) for g, v in raw.items()},
        "change_level_min": CHANGE_MIN,
        "status_boundaries": {g: tuple(np.round(np.array(v) - v[0] - GRID, 2))
                              for g, v in raw.items()},
        "change_boundaries": tuple(np.round(np.array(CHANGE_MIN) - GRID, 2)),
        "status_centre": {g: v[0] for g, v in raw.items()},
        "cut_set": grp,
        "unit_status_level_min": _status_cuts_c(df),
        "status_offset": df.status_offset.to_numpy(float),
    }


def _extras(df: pd.DataFrame) -> dict:
    t = np.where(df.hs == 1, "hs", "k8")
    group = (df.year.astype(str) + "-" + t).to_numpy()
    return {
        "group": group, "year": df.year.to_numpy(int), "cds": df.cds.to_numpy(str),
        "county": df.county.to_numpy(str), "district": df.district.to_numpy(str),
        "status": df.status.to_numpy(float), "change": df.change.to_numpy(float),
        "status_c": df.status_c.to_numpy(float),
        "prior_status": df.prior_c.to_numpy(float),
        "prior_status_published": (df.prior_published - df.status_offset).to_numpy(float),
        "n_curr": df.n_curr.to_numpy(float), "n_prior": df.n_prior.to_numpy(float),
        "statuslevel_published": _int(df.statuslevel),
        "changelevel_published": _int(df.changelevel),
        "color_published": _int(df.color),
        "red_published": (df.color == "1").to_numpy(float),
        "red_orange_published": df.color.isin(["1", "2"]).to_numpy(float),
        "y_year": df.y_year.to_numpy(int),
        "y_color": pd.to_numeric(df.y_color, errors="coerce").to_numpy(float),
        "charter": df.charter.to_numpy(float),
    }


def _two_score(subject: str, level: str = "red", cut_set: str = "all") -> RDDSample:
    df = _frame(subject, cut_set=cut_set)
    X, names = _design(df, prior=False)
    cuts = _status_cuts_c(df)
    ex = _extras(df)
    ex.update(_matrix_extras(subject, df))
    ex.update({"level": level, "treated_colors": [COLORS[c] for c in TREATED_COLORS[level]],
               "published_D": ex[f"{level}_published"]})
    thr = (-GRID, CHANGE_MIN[2] - GRID)
    return RDDSample(
        Q=np.column_stack([df.status_c, df.change]).astype(float), X=X,
        Y=df.y_status.to_numpy(float), threshold=thr,
        name=f"ca_dashboard_{subject}_two_score" + ("" if level == "red" else f"_{level}")
             + ("" if cut_set == "all" else f"_{cut_set}"),
        feature_names=names,
        description=(f"California School Dashboard, {subject.upper()} academic indicator, "
                     f"schools (all students), {', '.join(map(str, MAIN_YEARS))}. Q = (status "
                     "minus the lowest Low status of the school's cut-score set, change); "
                     f"treatment = the 5 x 5 table gives {' or '.join(ex['treated_colors'])}"
                     + (" = 1{Q1 < -0.05} * 1{Q2 < 2.95}" if level == "red" else "")
                     + "; Y = next Dashboard year's status."),
        citation="California Department of Education, California School Dashboard academic "
                 "indicator data files 2017-2026 and Dashboard Technical Guides",
        treatment_rule=matrix_rule(level, cuts), extras=ex,
    )


def _single(subject: str, level: str = "red", cut_set: str = "all") -> RDDSample:
    df = _frame(subject, cut_set=cut_set)
    X, names = _design(df, prior=True)
    prior = df.prior_c.to_numpy(float)
    cuts = _status_cuts_c(df)
    two = matrix_rule(level, cuts)

    def rule(Q: np.ndarray) -> np.ndarray:
        q = np.asarray(Q, float).reshape(-1)
        return two(np.column_stack([q, q - prior]))

    ex = _extras(df)
    ex.update(_matrix_extras(subject, df))
    ex.update({"level": level, "treated_colors": [COLORS[c] for c in TREATED_COLORS[level]],
               "published_D": ex[f"{level}_published"]})
    if level == "red":
        # D = 1{Q < c_i}: the Red cutoff on centred status for unit i.
        ex["cutoff"] = np.minimum(-GRID, prior + CHANGE_MIN[2] - GRID)
    return RDDSample(
        Q=df.status_c.to_numpy(float), X=X, Y=df.y_status.to_numpy(float), threshold=-GRID,
        name=f"ca_dashboard_{subject}" + ("" if level == "red" else f"_{level}")
             + ("" if cut_set == "all" else f"_{cut_set}"),
        feature_names=names,
        description=(f"California School Dashboard, {subject.upper()} academic indicator, "
                     f"schools (all students), {', '.join(map(str, MAIN_YEARS))}. Q = status "
                     "minus the lowest Low status of the school's cut-score set; the prior "
                     "status is in X; treatment = the 5 x 5 table at (Q, Q - prior) gives "
                     f"{' or '.join(ex['treated_colors'])}"
                     + (" = 1{Q < min(-0.05, prior + 2.95)}" if level == "red" else "")
                     + "; Y = next Dashboard year's status."),
        citation="California Department of Education, California School Dashboard academic "
                 "indicator data files 2017-2026 and Dashboard Technical Guides",
        treatment_rule=rule, extras=ex,
    )


def load() -> RDDSample:
    """ELA, single score: Q = centred status, prior status in X, Red by the 5 x 5 table at
    (Q, Q - prior), i.e. Q < min(-0.05, prior + 2.95)."""
    return _single("ela")


def load_math() -> RDDSample:
    """Mathematics, the single-score view of ``load()``."""
    return _single("math")


def load_two_score(level: str = "red", cut_set: str = "all") -> RDDSample:
    """ELA, Q = (centred status, change), treatment = the 5 x 5 table gives Red
    (``level="red_orange"``: Red or Orange). ``cut_set``: "all", "k8" (grades 3-8 cut
    scores) or "hs" (grade 11 cut scores)."""
    return _two_score("ela", level, cut_set)


def load_math_two_score(level: str = "red", cut_set: str = "all") -> RDDSample:
    """Mathematics, as ``load_two_score()``."""
    return _two_score("math", level, cut_set)


def load_status_2022(subject: str = "ela") -> RDDSample:
    """2022, status only (no change, no colors): Q = centred status, D = 1{Q < -0.05}, a
    Very Low status level (the 2022 basis for Differentiated Assistance and CSI);
    Y = the 2023 status. Schools with a published status level, not DASS or
    pair-and-share."""
    p = build()
    m = (p.subject.eq(subject) & (p.year == 2022) & p.statuslevel.isin(list("12345"))
         & (p.dass == 0) & (p.pairshare == "") & p.y_status.notna())
    df = p[m].reset_index(drop=True)
    X, names = _design(df, prior=False)
    ex = _extras(df.assign(prior_c=np.nan, change=np.nan))
    for k in ["change", "prior_status", "prior_status_published", "n_prior",
              "changelevel_published", "color_published", "red_published",
              "red_orange_published"]:
        ex.pop(k)                               # no change and no colors in 2022
    ex.update({"status_offset": df.status_offset.to_numpy(float),
               "published_D": (df.statuslevel == "1").to_numpy(float)})
    return RDDSample(
        Q=df.status_c.to_numpy(float), X=X, Y=df.y_status.to_numpy(float), threshold=-GRID,
        name=f"ca_dashboard_{subject}_status_2022", feature_names=names,
        description=(f"California School Dashboard 2022, {subject.upper()}: Q = status minus "
                     "the lowest Low status; treatment = 1{Q < -0.05} (Very Low); "
                     "Y = 2023 status."),
        citation="California Department of Education, California School Dashboard 2022 "
                 "academic indicator data files",
        treatment_rule=lambda Q: (np.asarray(Q, float) < -GRID).astype(int), extras=ex,
    )
