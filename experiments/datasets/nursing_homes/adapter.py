"""CMS Nursing Home Five-Star ratings: state-specific percentile cut points on the
health-inspection score, and a decision matrix on two case-mix adjusted staffing measures.

Health-inspection rating (Five-Star Technical Users' Guide, every version 2012-2020).
Each deficiency on the last three standard surveys (and on complaint surveys of the last
36 months) gets points by scope and severity; revisits add points. The cycles are weighted
1/2, 1/3, 1/6 (60/40 with two) into the weighted health-inspection score x (lower is
better). Within each state, the score's percentiles P10, P33.33, P56.67, P80 give 5 stars
(x <= P10), 4, 3, 2 (x <= P80) and 1 star (x > P80). CMS recomputes the cut points every
month, but "the rating for a given facility is held constant until there is a change in
the weighted health inspection score". In the data a change is a change in x or a new (or
re-dated) cycle-1 survey, and every facility was re-rated in 2018-02 and 2019-04. So a
rating follows the cut points of the month tau of the facility's last change:

    load():         Q = log(1 + c2) - log(1 + x),   D = 1{Q >= 0}  (2+ stars, not 1 star)
    load_points():  Q = c2 - x (points)

with c2 = CMS's published P80 for the state and month tau (``cutpoints.py``). The unit is
an inspection event: a facility-month in which a new standard survey becomes cycle 1, so
the facility is re-rated that month. After a month missing from the mirror tau is one of
two months (see README).

Freeze. Surveys on or after 2017-11-28 (the new survey process) were not used from data
month 2018-02 to 2019-03: scores were recomputed in 2018-02 from the two prior standard
surveys and frozen, and recomputed in 2019-04 from the three latest. Events are new
surveys entering in data months 2013-09 to 2018-01 and 2019-04 to 2019-10; 2019-04
(``extras["freeze_return"]``) re-scored every facility.

Staffing rating. RN and total nurse hours per resident day, case-mix adjusted and rounded
to three decimals, get 1-5 stars from national cut points; a table combines them. The
5-star cell is a both rule in every period:
    data months <= 2018-03 (CMS-671 staffing): RN >= 0.710 and total >= 4.418
    2018-04 to 2019-03 (PBJ):                  RN >= 0.884 and total >= 4.238
    from 2019-04:                              RN >= 1.049 and total >= 4.038 (RN 5, total 4+)
``load_two_score()``: Q = (RN - c_RN, total - c_total), D = 1{both >= 0}, staffing 5 stars.

Y (all loaders): log(1 + total health-inspection score of the facility's next standard
survey), as first published (lower is better); the survey is held after the event month.
``*_residents``: log residents at that next survey minus log residents at the event.

X: certified beds, occupancy, ownership, chain (an owning or managing organization shared
with another facility, from that year's ownership file), in a hospital, CCRC, Medicare /
Medicaid certification, resident and family councils, change of ownership in the last
12 months, years certified, urban (county in a CBSA), case-mix nurse hours (a resident-
acuity index), state and year indicators. Not the scores, their cycles, complaints, fines
or staffing hours. ``*_lagged`` add the score (or adjusted hours) of the month before.
"""
from __future__ import annotations

import gzip
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample
from experiments.datasets.nursing_homes import cutpoints
from experiments.datasets.nursing_homes.download import OWNERSHIP_TAGS, RAW, SNAP, SNAPSHOTS

HERE = Path(__file__).parent
PROCESSED = HERE / "data" / "processed"
PANEL = PROCESSED / "nursing_homes_panel.parquet"
EVENTS = PROCESSED / "nursing_homes_events.parquet"
CROSSWALK = RAW / "crosswalk" / "ssa_fips_state_county2016.csv"

FREEZE = ("2018-02", "2019-03")      # data months with frozen health-inspection scores
RETURN = "2019-04"                   # scores recomputed with all surveys (guide, April 2019)
TOL = 5e-4                           # scores and cut points are published to 0.001
# Staffing: national cut points of the 5-star cell (adjusted hours per resident day,
# rounded to 0.001), by data month: (first month, RN cut, total cut).
STAFF5 = (("2008-01", 0.710, 4.418), ("2018-04", 0.884, 4.238), ("2019-04", 1.049, 4.038))
# The full staffing tables, for rule_check(): component cut points (2, 3, 4, 5 stars) and
# the combination table (rows RN 1-5, columns total 1-5).
_T12 = [[1, 1, 2, 2, 3], [1, 2, 3, 3, 4], [2, 3, 4, 4, 4], [2, 3, 4, 4, 4], [3, 4, 4, 4, 5]]
_T15 = [[1, 1, 2, 2, 3], [1, 2, 3, 3, 3], [2, 3, 3, 4, 4], [2, 3, 4, 4, 4], [3, 3, 4, 4, 5]]
_T19 = [[1, 1, 2, 2, 3], [2, 2, 2, 3, 3], [2, 3, 3, 3, 4], [3, 3, 4, 4, 4], [3, 4, 4, 5, 5]]
STAFF_TABLES = (  # (first data month, RN cuts, total cuts, table, guide)
    ("2012-04", (0.283, 0.379, 0.513, 0.710), (3.262, 3.661, 4.173, 4.418), _T12, "July 2012"),
    ("2015-02", (0.283, 0.379, 0.513, 0.710), (3.262, 3.661, 4.173, 4.418), _T15, "Feb 2015"),
    ("2018-04", (0.246, 0.383, 0.587, 0.884), (3.176, 3.552, 4.010, 4.238), _T15, "Apr 2018"),
    ("2019-04", (0.317, 0.508, 0.731, 1.049), (3.108, 3.580, 4.038, 4.408), _T19, "Apr 2019"),
)

# lower-case column -> name used here. The 2017-08 snapshot spells out the column names.
_RENAME = {
    "exp_aide": "cm_aide", "exp_lpn": "cm_lpn", "exp_rn": "cm_rn", "exp_total": "cm_total",
    "sffstatus": "sff",
    "federal provider number": "provnum", "provider name": "provname",
    "provider state": "state", "provider ssa county code": "county_ssa",
    "provider county name": "county_name", "ownership type": "ownership",
    "number of certified beds": "bedcert", "number of residents in certified beds": "restot",
    "provider type": "certification", "provider resides in hospital": "inhosp",
    "date first approved to provide medicare and medicaid services": "participation_date",
    "continuing care retirement community": "ccrc_facil", "special focus facility": "sff",
    "most recent health inspection more than 2 years ago": "oldsurvey",
    "provider changed ownership in last 12 months": "chow_last_12mos",
    "with a resident and family council": "resfamcouncil",
    "overall rating": "overall_rating", "overall rating footnote": "overall_rating_fn",
    "health inspection rating": "survey_rating",
    "health inspection rating footnote": "survey_rating_fn",
    "qm rating": "quality_rating", "qm rating footnote": "quality_rating_fn",
    "staffing rating": "staffing_rating", "staffing rating footnote": "staffing_rating_fn",
    "rn staffing rating": "rn_staffing_rating",
    "rn staffing rating footnote": "rn_staffing_rating_fn",
    "reported staffing footnote": "staffing_flag",
    "physical therapist staffing footnote": "pt_staffing_flag",
    "reported rn staffing hours per resident per day": "rnhrd",
    "reported total nurse staffing hours per resident per day": "tothrd",
    "expected rn staffing hours per resident per day": "cm_rn",
    "expected total nurse staffing hours per resident per day": "cm_total",
    "adjusted rn staffing hours per resident per day": "adj_rn",
    "adjusted total nurse staffing hours per resident per day": "adj_total",
    "total weighted health survey score": "weighted_all_cycles_score",
    "number of facility reported incidents": "incident_cnt",
    "number of substantiated complaints": "cmplnt_cnt", "number of fines": "fine_cnt",
    "total amount of fines in dollars": "fine_tot",
    "number of payment denials": "payden_cnt", "total number of penalties": "tot_penlty_cnt",
    "processing date": "filedate",
    **{f"cycle {k} {a}": f"cycle_{k}_{b}" for k in (1, 2, 3) for a, b in (
        ("total number of health deficiencies", "defs"),
        ("number of standard health deficiencies", "nfromdefs"),
        ("number of complaint health deficiencies", "nfromcomp"),
        ("health deficiency score", "defs_score"),
        ("standard survey health date", "survey_date"),
        ("standard health survey date", "survey_date"),
        ("number of health revisits", "numrevis"), ("health revisit score", "revisit_score"),
        ("total health score", "total_score"))},
}
_KEEP = [
    "provnum", "provname", "state", "county_ssa", "county_name", "ownership", "bedcert",
    "restot", "certification", "inhosp", "participation_date", "ccrc_facil", "sff",
    "abuse", "oldsurvey", "chow_last_12mos", "resfamcouncil", "overall_rating",
    "overall_rating_fn", "survey_rating", "survey_rating_fn", "quality_rating",
    "quality_rating_fn", "staffing_rating", "staffing_rating_fn", "rn_staffing_rating",
    "rn_staffing_rating_fn", "staffing_flag", "rnhrd", "tothrd", "cm_rn", "cm_total",
    "adj_rn", "adj_total", "weighted_all_cycles_score", "cmplnt_cnt", "fine_cnt", "filedate",
    *[f"cycle_{k}_{v}" for k in (1, 2, 3) for v in (
        "survey_date", "defs", "nfromdefs", "nfromcomp", "defs_score", "numrevis",
        "revisit_score", "total_score")],
]
_NUM = ["bedcert", "restot", "overall_rating", "survey_rating", "quality_rating",
        "staffing_rating", "rn_staffing_rating", "rnhrd", "tothrd", "cm_rn", "cm_total",
        "adj_rn", "adj_total", "weighted_all_cycles_score", "cmplnt_cnt", "fine_cnt",
        *[f"cycle_{k}_{v}" for k in (1, 2, 3) for v in (
            "defs", "nfromdefs", "nfromcomp", "defs_score", "numrevis", "revisit_score",
            "total_score")]]
_DATES = ["participation_date", "filedate", "cycle_1_survey_date", "cycle_2_survey_date",
          "cycle_3_survey_date"]
# Staffing footnotes that mark a 1-star rating by exception (no data submitted, days
# without an RN, failed audit), not by the table.
_STAFF_EXCEPTION_FN = {"12", "No Staffing Data Submitted", "Downgraded to 1 Star"}


def _require(*paths: Path) -> None:
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(
                f"{p} missing. Run python -m experiments.datasets.nursing_homes.download")


def _write_parquet(df: pd.DataFrame, dest: Path) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=PROCESSED, prefix=".tmp_", suffix=".parquet")
    os.close(fd)
    try:
        df.to_parquet(tmp, index=False)
        os.chmod(tmp, 0o644)
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _date(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    out = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    return out.fillna(pd.to_datetime(s, format="%m/%d/%Y", errors="coerce"))


def _month(ts: pd.Series) -> pd.Series:
    return ts.dt.strftime("%Y-%m")


def _mnum(m: pd.Series) -> pd.Series:
    """'YYYY-MM' -> months since year 0."""
    m = m.astype(str)
    return m.str[:4].astype(int) * 12 + m.str[5:7].astype(int) - 1


def _mstr(n: pd.Series) -> pd.Series:
    n = n.astype(int)
    return (n // 12).astype(str) + "-" + (n % 12 + 1).astype(str).str.zfill(2)


# ----------------------------------------------------------------------------- panel

def read_snapshot(tag: str) -> pd.DataFrame:
    """ProviderInfo of one NBER snapshot, with harmonized column names and types."""
    path = SNAP / tag / "ProviderInfo_Download.csv.gz"
    _require(path)
    with gzip.open(path, "rb") as f:
        d = pd.read_csv(f, dtype=str, encoding="latin-1", keep_default_na=False)
    d.columns = [_RENAME.get(c.strip().lower(), c.strip().lower()) for c in d.columns]
    for c in _KEEP:
        if c not in d.columns:
            d[c] = ""
    d = d[_KEEP].copy()
    for c in d.columns:
        d[c] = d[c].str.strip()
    d["provnum"] = d.provnum.str.zfill(6)
    d["county_ssa"] = d.county_ssa.str.zfill(3)
    for c in _NUM:
        d[c] = pd.to_numeric(d[c].str.replace(",", "", regex=False), errors="coerce")
    for c in _DATES:
        d[c] = _date(d[c])
    d["sff_candidate"] = (d.sff == "SFF Candidate").astype(float)
    d["sff"] = d.sff.isin(["Y", "SFF"]).astype(float)
    for c in ["inhosp", "ccrc_facil", "abuse", "oldsurvey", "chow_last_12mos"]:
        d[c] = d[c].str.upper().str[:1].map({"Y": 1.0, "N": 0.0})
    d["tag"] = tag
    return d


def build_panel() -> pd.DataFrame:
    """Every facility in every snapshot (one row per facility x NBER folder), cached."""
    if not PANEL.exists():
        tags = [t for t in SNAPSHOTS if (SNAP / t / "ProviderInfo_Download.csv.gz").exists()]
        if not tags:
            _require(SNAP / "2015_07" / "ProviderInfo_Download.csv.gz")
        _write_parquet(pd.concat([read_snapshot(t) for t in tags], ignore_index=True), PANEL)
    return pd.read_parquet(PANEL)


def monthly_panel() -> pd.DataFrame:
    """One row per facility x data month (FILEDATE). Snapshots that repeat an earlier data
    month are dropped (the first NBER folder with that FILEDATE is kept)."""
    p = build_panel()
    p = p[p.filedate.notna()].copy()
    p["month"] = _month(p.filedate)
    first = p.groupby("month").tag.min()
    p = p[p.tag == p.month.map(first)].copy()
    p = p.sort_values(["provnum", "month"]).drop_duplicates(["provnum", "month"])
    return p.reset_index(drop=True)


# ------------------------------------------------------------------------- the rules

def stars(score, c5, c4, c3, c2) -> np.ndarray:
    """Health-inspection stars from a weighted score and the state's cut points (all on
    CMS's 0.001 grid): 5 if x <= P10, 4 if <= P33, 3 if <= P57, 2 if <= P80, else 1."""
    x = np.round(np.asarray(score, float), 3)
    cs = [np.round(np.asarray(c, float), 3) + TOL for c in (c5, c4, c3, c2)]
    out = np.select([x <= cs[0], x <= cs[1], x <= cs[2], x <= cs[3]], [5, 4, 3, 2], 1)
    return np.where(np.isnan(x) | np.isnan(cs[3]), np.nan, out)


def staffing_stars(adj_rn, adj_total, month) -> tuple:
    """(RN stars, total stars, staffing stars) from the guide's table for each data month."""
    rn = np.round(np.asarray(adj_rn, float), 3)
    tot = np.round(np.asarray(adj_total, float), 3)
    month = np.asarray(month, str)
    r = np.full(len(rn), np.nan)
    t = np.full(len(rn), np.nan)
    s = np.full(len(rn), np.nan)
    for i, (start, rc, tc, tab, _) in enumerate(STAFF_TABLES):
        end = STAFF_TABLES[i + 1][0] if i + 1 < len(STAFF_TABLES) else "9999-99"
        m = (month >= start) & (month < end) & ~np.isnan(rn) & ~np.isnan(tot)
        rr = 1 + sum((rn[m] >= c - 1e-9).astype(int) for c in rc)
        tt = 1 + sum((tot[m] >= c - 1e-9).astype(int) for c in tc)
        r[m], t[m] = rr, tt
        s[m] = np.asarray(tab)[rr - 1, tt - 1]
    return r, t, s


def staff5_cuts(month) -> tuple:
    month = np.asarray(month, str)
    rc = np.full(len(month), np.nan)
    tc = np.full(len(month), np.nan)
    for start, a, b in STAFF5:
        m = month >= start
        rc[m], tc[m] = a, b
    return rc, tc


def overall_stars(insp, staff, qm, sff=None) -> np.ndarray:
    """Overall rating from the three domain ratings (guide, every version 2012-2019); an
    SFF that has not graduated gets at most 3 stars (guides before April 2019)."""
    h = np.asarray(insp, float)
    s = np.asarray(staff, float)
    q = np.asarray(qm, float)
    o = h.copy()
    s_ok = ~np.isnan(s)
    o = np.where(s_ok & (s >= 4) & (s > h), o + 1, o)
    o = np.where(s_ok & (s == 1), o - 1, o)
    o = np.clip(o, 1, 5)
    q_ok = ~np.isnan(q)
    o = np.where(q_ok & (q == 5), o + 1, o)
    o = np.where(q_ok & (q == 1), o - 1, o)
    o = np.clip(o, 1, 5)
    o = np.where(h == 1, np.minimum(o, 2), o)      # 1-star inspection: at most +1
    if sff is not None:
        o = np.where(np.asarray(sff, float) == 1, np.minimum(o, 3), o)
    return o


# ------------------------------------------------------------------ rating months, events

def _with_history(p: pd.DataFrame) -> pd.DataFrame:
    """Adds, per facility-month: the previous observed month, whether the weighted score
    changed, whether a new standard survey entered (cycle 1 date changed), and the window
    (tau_lo, tau_hi) of data months in which the current score first applied."""
    p = p.sort_values(["provnum", "month"]).copy()
    p["score"] = p.weighted_all_cycles_score.round(3)
    p["mn"] = _mnum(p.month)
    g = p.groupby("provnum", sort=False)
    p["prev_mn"] = g.mn.shift()
    p["prev_score"] = g.score.shift()
    p["prev_c1"] = g.cycle_1_survey_date.shift()
    p["prev_rating"] = g.survey_rating.shift()
    p["prev_adj_rn"] = g.adj_rn.shift()
    p["prev_adj_total"] = g.adj_total.shift()
    p["gap"] = p.mn - p.prev_mn
    first = p.prev_mn.isna()

    def moved(cur: pd.Series, prev: pd.Series, tol: float) -> pd.Series:
        a, b = cur.astype(float), prev.astype(float)
        return ~(((a - b).abs() < tol) | (a.isna() & b.isna()))

    p["score_changed"] = ~first & moved(p.score, p.prev_score, TOL)
    p["new_survey"] = (~first & p.cycle_1_survey_date.notna() & p.prev_c1.notna()
                       & (p.cycle_1_survey_date != p.prev_c1))
    # CMS re-rates a facility when its weighted score changes, and also when a new (or
    # re-dated) standard survey becomes cycle 1 although the score happens to stay the
    # same; other changes inside the cycles with an unchanged score keep the rating (see
    # README). In 2018-02 (start of the freeze) and 2019-04 (its end) every facility was
    # re-scored and re-rated with that month's cut points.
    p["changed"] = ~first & (p.score_changed | p.new_survey
                             | p.month.isin([FREEZE[0], RETURN]))
    # Runs without a change: tau is in (previous observed month, run start].
    run = (first | p.changed).astype(int).groupby(p.provnum).cumsum()
    key = p.provnum + "_" + run.astype(str)
    p["tau_hi"] = p.groupby(key).mn.transform("first")
    p["tau_lo"] = p.groupby(key).prev_mn.transform("first") + 1
    # The guide dates the re-scoring of every facility (2019-04, across the missing 2019-03).
    for mth in (FREEZE[0], RETURN):
        mm = int(_mnum(pd.Series([mth])).iloc[0])
        p.loc[(p.tau_hi == mm) & p.tau_lo.notna(), "tau_lo"] = mm
    p["tau_known"] = p.tau_lo == p.tau_hi        # NaN tau_lo (left-censored) -> False
    return p


def _attach_cuts(p: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    cp = cutpoints.table()[["month", "state", "c5", "c4", "c3", "c2"]]
    key = p[[col, "state"]].copy()
    key["month"] = np.where(key[col].notna(), _mstr(key[col].fillna(0)), "")
    m = key[["month", "state"]].merge(cp, on=["month", "state"], how="left")
    for c in ["c5", "c4", "c3", "c2"]:
        p[f"{prefix}{c}"] = m[c].to_numpy()
    return p


def _chain() -> pd.DataFrame:
    """Per ownership file (first snapshot of each year): a facility's chain size, the
    largest number of facilities linked to one of its organizational owners or managers
    (5%+ direct or indirect ownership, partnership interest, operational or managerial
    control)."""
    roles = ("OWNERSHIP INTEREST", "PARTNERSHIP INTEREST", "OPERATIONAL/MANAGERIAL CONTROL")
    out = []
    for tag in OWNERSHIP_TAGS:
        path = SNAP / tag / "Ownership_Download.csv.gz"
        if not path.exists():
            continue
        d = pd.read_csv(path, dtype=str, encoding="latin-1", keep_default_na=False)
        d.columns = [c.strip().lower() for c in d.columns]
        if "org_name" in d.columns:                      # 2013 layout
            org = d.org_name.str.strip()
        else:
            org = d.owner_name.str.strip().where(d.owner_type.str.strip() == "Organization", "")
        d = pd.DataFrame({"provnum": d.provnum.str.strip().str.zfill(6),
                          "org": org.str.upper(), "role": d.role_desc.str.upper()})
        d = d[(d.org != "") & d.role.str.contains("|".join(roles), regex=True)]
        d = d.drop_duplicates(["provnum", "org"])
        size = d.groupby("org").provnum.nunique()
        d["size"] = d.org.map(size)
        cs = d.groupby("provnum")["size"].max().rename("chain_size").reset_index()
        cs["year"] = int(tag[:4])
        out.append(cs)
    return pd.concat(out, ignore_index=True)


def _urban() -> pd.DataFrame:
    _require(CROSSWALK)
    x = pd.read_csv(CROSSWALK, dtype=str, keep_default_na=False)
    x["county_ssa"] = x.ssacounty.str[-3:]
    x["urban"] = ((x.cbsa.str.strip() != "") & ~x.cbsa.str.startswith("999")).astype(float)
    return x[["state", "county_ssa", "urban"]].drop_duplicates(["state", "county_ssa"])


def _surveys(p: pd.DataFrame) -> pd.DataFrame:
    """Every standard survey seen in cycles 1-3 of any month: its total score and the
    residents at its first publication."""
    s = pd.concat([p[["provnum", "mn", "restot", f"cycle_{k}_survey_date",
                      f"cycle_{k}_total_score"]].set_axis(
        ["provnum", "mn", "restot", "date", "total"], axis=1).assign(cycle=k)
        for k in (1, 2, 3)], ignore_index=True)
    s = s[s.date.notna()].sort_values(["provnum", "date", "mn", "cycle"])
    return s.drop_duplicates(["provnum", "date"]).rename(columns={"mn": "first_mn"})


def build_events() -> pd.DataFrame:
    """Facility-months in which a new standard survey enters the health-inspection score,
    with the deployed cut points, the rule, outcomes and covariates (cached)."""
    if EVENTS.exists():
        return pd.read_parquet(EVENTS)
    p = _with_history(monthly_panel())
    p = _attach_cuts(p, "tau_hi", "hi_")
    p = _attach_cuts(p, "tau_lo", "lo_")
    ev = p[p.new_survey].copy()
    # tau: the month the rating was set. After a missing snapshot month it is one of two
    # months; the published five-level rating picks it when the two months' cut points
    # give different stars, otherwise the later month is used (tau_ambiguous: same stars,
    # Q differs by the cut points' monthly drift).
    cuts = ["c5", "c4", "c3", "c2"]
    s_hi = stars(ev.score, *[ev[f"hi_{c}"] for c in cuts])
    s_lo = stars(ev.score, *[ev[f"lo_{c}"] for c in cuts])
    ev["tau_window"] = ev.tau_hi - ev.tau_lo + 1
    two = ev.tau_window > 1
    use_lo = (two & (s_hi != ev.survey_rating) & (s_lo == ev.survey_rating)).to_numpy()
    ev["tau"] = np.where(use_lo, ev.tau_lo, ev.tau_hi)
    for c in cuts:
        ev[c] = np.where(use_lo, ev[f"lo_{c}"], ev[f"hi_{c}"])
    ev["tau_ambiguous"] = (two & (s_hi == s_lo)).astype(float)
    ev["tau_from_rating"] = (two & (s_hi != s_lo)).astype(float)
    ev["stars_rule_hi"], ev["stars_rule_lo"] = s_hi, s_lo
    # The next standard survey after the event's cycle-1 survey.
    sv = _surveys(p)
    nxt = ev[["provnum", "cycle_1_survey_date"]].reset_index().merge(
        sv, on="provnum", how="left")
    nxt = nxt[nxt.date > nxt.cycle_1_survey_date].sort_values(["index", "date"])
    nxt = nxt.drop_duplicates("index").set_index("index")
    ev["next_date"] = nxt.date
    ev["next_mn"] = nxt.first_mn
    ev["next_cycle"] = nxt.cycle
    ev["y_next_score"] = nxt.total
    ev["y_log_next_score"] = np.log1p(nxt.total)
    ev["next_restot"] = nxt.restot
    ev["y_log_residents"] = np.log(ev.next_restot.where(ev.next_restot > 0)) - np.log(
        ev.restot.where(ev.restot > 0))
    # Weighted score 12 months later (the next period's rating input), if observed.
    later = p[["provnum", "mn", "score"]].rename(columns={"mn": "mn12", "score": "score12"})
    ev["mn12"] = ev.mn + 12
    ev = ev.merge(later, on=["provnum", "mn12"], how="left")
    # Rule at the 1/2 boundary and the five levels, with the cut points of tau.
    ev["stars_rule"] = stars(ev.score, ev.c5, ev.c4, ev.c3, ev.c2)
    rn, tot, st = staffing_stars(ev.adj_rn, ev.adj_total, ev.month)
    ev["rn_stars_rule"], ev["total_stars_rule"], ev["staffing_rule"] = rn, tot, st
    ev["staff_exception"] = ev.staffing_rating_fn.isin(_STAFF_EXCEPTION_FN).astype(float)
    ev["year"] = ev.month.str[:4].astype(int)
    ch = _chain()
    ev = ev.merge(ch, on=["provnum", "year"], how="left")
    ev = ev.merge(_urban(), on=["state", "county_ssa"], how="left")
    ev = ev.reset_index(drop=True)
    _write_parquet(ev, EVENTS)
    return ev


# ------------------------------------------------------------------------------ checks

def _sas_pctl(x: np.ndarray, q: float) -> float:
    """SAS's default percentile (PCTLDEF=5): average of the two order statistics when
    n*q is an integer, else the next order statistic."""
    x = np.sort(x)
    n = len(x)
    j = int(np.floor(n * q + 1e-9))
    if abs(n * q - j) < 1e-9:
        return float((x[j - 1] + x[min(j, n - 1)]) / 2) if j > 0 else float(x[0])
    return float(x[j])


def reproduce_cutpoints(p: pd.DataFrame | None = None) -> pd.DataFrame:
    """The state percentiles recomputed from each month's scores (all facilities with a
    weighted score, rated or not), next to CMS's published cut points."""
    p = monthly_panel() if p is None else p
    p = p[p.weighted_all_cycles_score.notna()]
    rows = []
    for (m, s), g in p.groupby(["month", "state"]):
        x = g.weighted_all_cycles_score.round(3).to_numpy()
        rows.append({"month": m, "state": s, "n_data": len(x),
                     **{f"r{c}": _sas_pctl(x, q) for c, q in
                        (("c5", 0.10), ("c4", 0.3333), ("c3", 0.56667), ("c2", 0.80))}})
    r = pd.DataFrame(rows)
    return r.merge(cutpoints.table(), on=["month", "state"], how="inner")


def rule_check() -> dict:
    """The rules against the published ratings (see README, "Rule check")."""
    p = _with_history(monthly_panel())
    p = _attach_cuts(p, "mn", "cur_")
    p = _attach_cuts(p, "tau_hi", "")
    rated = p.survey_rating.notna() & p.score.notna() & p.cur_c2.notna()
    frz = p.month.between(*FREEZE)
    cur = stars(p.score, p.cur_c5, p.cur_c4, p.cur_c3, p.cur_c2)
    tau = stars(p.score, p.c5, p.c4, p.c3, p.c2)
    pub = p.survey_rating.to_numpy()
    out: dict = {"data_months": sorted(p.month.unique().tolist()),
                 "facility_months": int(len(p))}
    for lab, m in (("outside_freeze", rated & ~frz), ("freeze", rated & frz)):
        known = m & p.tau_known
        chg = known & (p.tau_hi == p.mn)
        abuse = (p.abuse == 1) & (p.month >= "2019-10")
        out[lab] = {
            "rated": int(m.sum()),
            "stars_current_month_cut": float((cur == pub)[m].mean()),
            "tau_known": int(known.sum()),
            "stars_tau_cut": float((tau == pub)[known].mean()),
            "stars_tau_cut_excl_abuse_cap": float((tau == pub)[known & ~abuse].mean()),
            "score_changed_this_month": int(chg.sum()),
            "stars_this_month_cut_if_changed": float((tau == pub)[chg].mean()),
            "one_star_boundary_tau_cut": float(((tau >= 2) == (pub >= 2))[known].mean()),
            "mismatch_tau_one_star_boundary": int(((tau >= 2) != (pub >= 2))[known].sum()),
        }
    # Staffing table (all facility-months with adjusted hours and a rating).
    rn, tot, st = staffing_stars(p.adj_rn, p.adj_total, p.month)
    exc = p.staffing_rating_fn.isin(_STAFF_EXCEPTION_FN)
    ok = p.staffing_rating.notna() & ~np.isnan(st)
    s5 = st == 5
    out["staffing"] = {
        "rated_with_hours": int(ok.sum()),
        "table_agree": float((st == p.staffing_rating)[ok].mean()),
        "table_agree_excl_exceptions": float((st == p.staffing_rating)[ok & ~exc].mean()),
        "rn_rating_agree_excl_exceptions": float((rn == p.rn_staffing_rating)[ok & ~exc].mean()),
        "exceptions_1_star": int((ok & exc).sum()),
        "mismatch_excl_exceptions": int(((st != p.staffing_rating) & ok & ~exc).sum()),
        "five_star_both_rule_agree": float((s5 == (p.staffing_rating == 5))[ok & ~exc].mean()),
        "by_table": {t[4]: float((st == p.staffing_rating)[ok & ~exc & (p.month >= t[0]) & (
            p.month < (STAFF_TABLES[i + 1][0] if i + 1 < len(STAFF_TABLES) else "9999"))].mean())
            for i, t in enumerate(STAFF_TABLES)},
    }
    # Overall rating from the published domain ratings.
    ov = overall_stars(p.survey_rating, p.staffing_rating, p.quality_rating, p.sff)
    okv = p.overall_rating.notna() & p.survey_rating.notna()
    out["overall"] = {"rated": int(okv.sum()),
                      "agree": float((ov == p.overall_rating)[okv].mean()),
                      "mismatch": int(((ov != p.overall_rating) & okv).sum())}
    # Cut points recomputed from the data.
    r = reproduce_cutpoints(p)
    big = r.n_fac >= 5
    out["cutpoints_reproduced"] = {
        "state_months": int(big.sum()),
        # 0.002: an average of two scores published to 0.001 can round either way.
        **{c: float((np.abs(r[f"r{c}"] - r[c]) < 0.002)[big].mean())
           for c in ("c5", "c4", "c3", "c2")},
        "n_matches_official": float((r.n_data == r.n_fac)[big].mean()),
    }
    return out


# ------------------------------------------------------------------------------ samples

_OWN = {"For profit": "forprofit", "Government": "government", "Non profit": "nonprofit"}


def _design(df: pd.DataFrame) -> tuple:
    X = pd.DataFrame(index=df.index)
    X["log_beds"] = np.log(df.bedcert.clip(lower=1))
    occ = (df.restot / df.bedcert).clip(0, 1.5)
    X["occupancy"] = occ.fillna(occ.median())
    X["occupancy_missing"] = occ.isna().astype(float)
    own = df.ownership.str.split(" - ").str[0].map(_OWN).fillna("nonprofit")
    X["forprofit"] = (own == "forprofit").astype(float)
    X["government"] = (own == "government").astype(float)
    sub = df.ownership.str.split(" - ").str[1].fillna("")
    for k, v in {"Corporation": "corp", "Partnership": "partnership",
                 "Individual": "individual", "Limited Liability company": "llc"}.items():
        X[f"forprofit_{v}"] = ((own == "forprofit") & (sub == k)).astype(float)
    X["church"] = (sub == "Church related").astype(float)
    cs = df.chain_size.fillna(1).clip(lower=1)
    X["chain"] = (cs >= 2).astype(float)
    X["log_chain_size"] = np.log(cs)
    X["chain_missing"] = df.chain_size.isna().astype(float)
    X["in_hospital"] = df.inhosp.fillna(0)
    X["ccrc"] = df.ccrc_facil.fillna(0)
    X["medicare_only"] = (df.certification == "Medicare").astype(float)
    X["medicaid_only"] = (df.certification == "Medicaid").astype(float)
    for k in ("Both", "Family", "None"):                      # resident council only: base
        X[f"council_{k.lower()}"] = (df.resfamcouncil == k).astype(float)
    X["chow_12m"] = df.chow_last_12mos.fillna(0)
    age = (df.filedate - df.participation_date).dt.days / 365.25
    X["log_years_certified"] = np.log1p(age.clip(lower=0).fillna(age.median()))
    X["urban"] = df.urban.fillna(0)
    X["urban_missing"] = df.urban.isna().astype(float)
    cm = df.cm_total.where(df.cm_total > 0)
    X["log_casemix_total"] = np.log(cm.fillna(cm.median()))
    X["casemix_rn_share"] = (df.cm_rn / cm).fillna((df.cm_rn / cm).median())
    X["casemix_missing"] = cm.isna().astype(float)
    for s in sorted(df.state.unique()):
        if s != "CA":                                        # California base
            X[f"state_{s}"] = (df.state == s).astype(float)
    for y in sorted(df.year.unique())[1:]:
        X[f"year_{y}"] = (df.year == y).astype(float)
    # The staffing measures changed at 2018-04 (PBJ, RUG-IV case mix): absorbed by years.
    X = X.loc[:, X.std() > 0]
    return X.to_numpy(float), list(X.columns)


def _y_after(ev: pd.DataFrame, y_col: str) -> pd.Series:
    """Y observed, from a next standard survey held after the rating was published (the
    first day of the event's data month)."""
    return ev[y_col].notna() & (ev.next_date >= pd.to_datetime(ev.month + "-01"))


def _inspection_frame(boundary: int, y_col: str = "y_log_next_score") -> pd.DataFrame:
    ev = build_events()
    m = (~ev.month.between(*FREEZE) & ev.survey_rating.notna() & ev.score.notna()
         & ev.c2.notna() & (ev.tau_window <= 2) & _y_after(ev, y_col))
    if boundary >= 3:                    # abuse icon caps the rating at 2 stars (2019-10 on)
        m &= ~((ev.abuse == 1) & (ev.month >= "2019-10"))
    return ev[m].reset_index(drop=True)


def _extras(df: pd.DataFrame) -> dict:
    return {
        "provnum": df.provnum.to_numpy(str), "state": df.state.to_numpy(str),
        "month": df.month.to_numpy(str), "year": df.year.to_numpy(int),
        "score": df.score.to_numpy(float),
        "cut_points": df[["c5", "c4", "c3", "c2"]].to_numpy(float),
        "cut_month": _mstr(df.tau).to_numpy(str),
        "tau_ambiguous": df.tau_ambiguous.to_numpy(float),
        "tau_from_rating": df.tau_from_rating.to_numpy(float),
        "survey_rating_published": df.survey_rating.to_numpy(float),
        "overall_rating_published": df.overall_rating.to_numpy(float),
        "staffing_rating_published": df.staffing_rating.to_numpy(float),
        "quality_rating_published": df.quality_rating.to_numpy(float),
        "cycle_1_survey_date": df.cycle_1_survey_date.dt.strftime("%Y-%m-%d").to_numpy(str),
        "cycle_1_total_score": df.cycle_1_total_score.to_numpy(float),
        "next_survey_date": df.next_date.dt.strftime("%Y-%m-%d").to_numpy(str),
        "next_survey_month": _mstr(df.next_mn.fillna(0)).to_numpy(str),
        "y_next_score": df.y_next_score.to_numpy(float),
        "y_log_next_score": df.y_log_next_score.to_numpy(float),
        "score_before": df.prev_score.to_numpy(float),
        "y_log_residents": df.y_log_residents.to_numpy(float),
        "score_12m_later": df.score12.to_numpy(float),
        "freeze_return": (df.month == RETURN).to_numpy(float),
        "gap_months": df.gap.to_numpy(float),
        "sff": df.sff.to_numpy(float), "abuse_icon": df.abuse.to_numpy(float),
        "chain_size": df.chain_size.to_numpy(float),
        "restot": df.restot.to_numpy(float), "bedcert": df.bedcert.to_numpy(float),
    }


_CITATION = ("CMS Nursing Home Compare downloadable database, monthly snapshots 2013-08 to "
             "2019-10 (NBER mirror, data.nber.org/compare/nursing-home); CMS Five-Star "
             "Quality Rating System Technical Users' Guide and State-Level Health Inspection "
             "Cut Point Tables (cms.gov, Internet Archive captures)")
_Y_LABEL = {"y_log_next_score": "log(1 + total health-inspection score of the next standard "
                                "survey) (lower is better)",
            "y_log_residents": "log residents at the next standard survey - log residents now"}
_BOUND = {2: ("c2", "1 star vs 2+", 80), 3: ("c3", "2 or fewer vs 3+", 56.67),
          4: ("c4", "3 or fewer vs 4+", 33.33), 5: ("c5", "4 or fewer vs 5", 10)}


def _lagged(X: np.ndarray, names: list, cols: dict) -> tuple:
    """Adds lagged values (the month before the event) to X, with a missing indicator."""
    extra = []
    for name, v in cols.items():
        v = pd.Series(np.asarray(v, float))
        extra += [v.fillna(v.median()).to_numpy(), v.isna().astype(float).to_numpy()]
        names = names + [name, f"{name}_missing"]
    return np.column_stack([X, *extra]), names


def _inspection(boundary: int = 2, y_col: str = "y_log_next_score", scale: str = "log",
                lagged: bool = False) -> RDDSample:
    df = _inspection_frame(boundary, y_col)
    X, names = _design(df)
    if lagged:
        X, names = _lagged(X, names, {"log1p_score_before": np.log1p(df.prev_score)})
    cut, lab, pct = _BOUND[boundary]
    c = df[cut].round(3).to_numpy(float)
    q_points = np.round(c - df.score.to_numpy(float), 3)
    q_log = np.log1p(c) - np.log1p(df.score.to_numpy(float))
    q_log = np.where(np.abs(q_points) < TOL, 0.0, q_log)       # ties at the cut point
    Q = q_log if scale == "log" else q_points
    ex = _extras(df)
    ex.update({
        "boundary": boundary, "cutoff": 0.0, "scale": scale,
        "q_points": q_points, "q_log": q_log,
        "eligible_published": (df.survey_rating >= boundary).to_numpy(float),
        "rule_stars": df.stars_rule.to_numpy(float),
        "treated_side": (f"Q >= 0: the weighted score x is at or below the state's published "
                         f"P{pct} cut point c for the month the rating was set ({boundary}+ "
                         f"stars, {lab}; treated = the better rating)"),
    })
    qdef = "log(1 + c) - log(1 + x)" if scale == "log" else "c - x (points)"
    sfx = ("" if y_col == "y_log_next_score" else "_residents") + (
        "" if scale == "log" else "_points") + ("_lagged" if lagged else "")
    return RDDSample(
        Q=Q, X=X, Y=df[y_col].to_numpy(float), threshold=0.0,
        name=f"nursing_homes_inspection_{boundary}star{sfx}", feature_names=names,
        description=(f"US nursing homes, CMS Five-Star health-inspection rating at inspection "
                     f"events 2013-2019. Q = {qdef}, x the weighted deficiency score and c the "
                     f"state's P{pct} cut point of the month the rating was set; treatment = "
                     f"1{{Q >= 0}}, {boundary}+ stars; Y = {_Y_LABEL[y_col]}."),
        citation=_CITATION, treatment_rule=lambda q: (q >= 0).astype(int), extras=ex,
    )


def _staffing(y_col: str = "y_log_next_score", lagged: bool = False) -> RDDSample:
    ev = build_events()
    m = (ev.adj_rn.notna() & ev.adj_total.notna() & ev.staffing_rating.notna()
         & (ev.staff_exception == 0) & _y_after(ev, y_col) & ~ev.month.between(*FREEZE))
    df = ev[m].reset_index(drop=True)
    X, names = _design(df)
    if lagged:
        X, names = _lagged(X, names, {"adj_rn_before": df.prev_adj_rn,
                                      "adj_total_before": df.prev_adj_total})
    rc, tc = staff5_cuts(df.month)
    Q = np.column_stack([np.round(df.adj_rn.round(3) - rc, 3),
                         np.round(df.adj_total.round(3) - tc, 3)])
    ex = _extras(df)
    ex.update({
        "cutoffs": (0.0, 0.0), "rule": "both",
        "cut_rn": rc, "cut_total": tc,
        "adj_rn": df.adj_rn.round(3).to_numpy(float),
        "adj_total": df.adj_total.round(3).to_numpy(float),
        "eligible_published": (df.staffing_rating == 5).to_numpy(float),
        "staffing_rule_stars": df.staffing_rule.to_numpy(float),
        "rn_stars_rule": df.rn_stars_rule.to_numpy(float),
        "total_stars_rule": df.total_stars_rule.to_numpy(float),
        "staff5_periods": STAFF5,
        "staffing_tables": {t[4]: {"from": t[0], "rn_cuts": t[1], "total_cuts": t[2],
                                   "table_rn_by_total": t[3]} for t in STAFF_TABLES},
        "treated_side": ("staffing 5 stars: Q1 >= 0 and Q2 >= 0 (adjusted RN and total nurse "
                         "hours per resident day, rounded to 0.001, at or above the 5-star "
                         "cut points of the data month)"),
    })
    sfx = ("" if y_col == "y_log_next_score" else "_residents") + ("_lagged" if lagged else "")
    return RDDSample(
        Q=Q, X=X, Y=df[y_col].to_numpy(float), threshold=(0.0, 0.0),
        name=f"nursing_homes_staffing_5star{sfx}", feature_names=names,
        description=("US nursing homes, CMS Five-Star staffing rating at inspection events "
                     "2013-2019. Q = (adjusted RN hours - RN cut, adjusted total nurse hours "
                     "- total cut), the 5-star cell of the staffing table; treatment = "
                     f"1{{Q1 >= 0 and Q2 >= 0}}; Y = {_Y_LABEL[y_col]}."),
        citation=_CITATION,
        treatment_rule=lambda q: ((q[:, 0] >= -1e-9) & (q[:, 1] >= -1e-9)).astype(int),
        extras=ex,
    )


def load() -> RDDSample:
    """Health-inspection rating, 1 star vs 2+ stars. Q = log(1 + c2) - log(1 + score), c2 the
    state's P80 cut point of the month the rating was set; D = 1{Q >= 0} (2+ stars);
    Y = log(1 + next standard survey's score)."""
    return _inspection(2)


def load_points() -> RDDSample:
    """As ``load()``, with Q = c2 - score in deficiency points."""
    return _inspection(2, scale="points")


def load_lagged() -> RDDSample:
    """As ``load()``, with X + log(1 + the weighted score the month before the event)."""
    return _inspection(2, lagged=True)


def load_residents() -> RDDSample:
    """As ``load()``, with Y = log residents at the next survey - log residents now."""
    return _inspection(2, "y_log_residents")


def load_23() -> RDDSample:
    return _inspection(3)


def load_34() -> RDDSample:
    return _inspection(4)


def load_45() -> RDDSample:
    return _inspection(5)


def load_two_score() -> RDDSample:
    """Staffing 5 stars: Q = (RN - c_RN, total - c_total); D = 1{both >= 0}; Y = log(1 +
    next standard survey's score)."""
    return _staffing()


def load_two_score_lagged() -> RDDSample:
    """As ``load_two_score()``, with X + the adjusted hours the month before the event."""
    return _staffing(lagged=True)


def load_two_score_residents() -> RDDSample:
    return _staffing("y_log_residents")
