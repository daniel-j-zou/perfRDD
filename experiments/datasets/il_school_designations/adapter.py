"""Illinois ESSA summative designations (ISBE): percentile cutoffs on a weighted index.

Each Illinois public school with enough data gets, per grade band (ES = grades 1-8, HS =
grades 9-12), an index score: a weighted sum of indicator scores (0-100) for the "All
Students" group and for each student group with n >= 20 (Summative Designation Business
Rules, Processes 3-4; the 2023, 2024 and 2025 editions agree):

    ES: 7.5 ELA proficiency + 7.5 math proficiency + 25 ELA growth + 25 math growth
        + 5 science proficiency + 5 EL progress to proficiency (ELPtP)        [academic 75]
        + 20 chronic absenteeism + 5 climate survey participation              [SQSS 25]
    HS: 7.5 ELA proficiency + 7.5 math proficiency + 50 graduation rate
        + 5 science proficiency + 5 ELPtP                                      [academic 75]
        + 10 chronic absenteeism + 6.67 climate survey + 8.33 9th grade on track [SQSS 25]

all divided by 100, with the weight of a missing indicator spread over the present
indicators of its category in proportion to their weights. Process 5 ranks the
All-Students index scores of each band (a school serving both bands is in both lists);
the Exemplary threshold T10 is the score at the top-10% rank and the Comprehensive
threshold T5 the score at the bottom-5% rank (the 2025 rules: ranks round(0.1 n) and
round(0.95 n) + 1). Then, in order:

    Comprehensive  index <= T5 ("All Students"); in HS also a graduation-rate indicator
                   score of 0, i.e. a composite (0.6 x 4-year + 0.3 x 5-year + 0.1 x 6-year)
                   graduation rate <= 66.67% ("Low Graduation Rate");
    Targeted       any student group's index <= T5;
    Exemplary      index >= T10;
    Commendable    otherwise.

Comprehensive schools enter IL-EMPOWER (a four-year improvement cycle with state funding).
A Comprehensive school in year 3 of its grant is published as "Intensive", and a Targeted
school in year 3 whose group is still at or below T5 is published as Comprehensive ("From
Targeted"); both follow from ISBE's funding tables, which are not public.

The adapter takes T5 and T10 at full precision from ISBE's state SummativeSchoolDistribution
report (the Descriptive Statistics PDFs round them to 0.01). The ranks cannot be recomputed
exactly from public data: the service publishes no elementary-band index for the 77-87
dually designated schools a year, and in 2024-2025 the high-school thresholds sit one rank
from round(0.1 n) / round(0.95 n) + 1. With ISBE's thresholds the rule reproduces every
published designation, 2023-2025 (``rule_check``).

Q = index - (T5 + 0.00005) in ``load()``: the published index (the "Overall" row, four
    decimals) minus a cutoff half a grid step above that year-and-band's T5 (rounded to
    four decimals), so that D = 1{Q >= 0} = index > T5 = not Comprehensive by the index
    rule (the untreated side is Comprehensive). ``load_exemplary()``: Q = index - T10,
    D = 1{Q >= 0} = Exemplary. ``load_two_score()``: high schools, Q = (index - (T5 +
    0.00005), composite graduation rate), D = 1{Q1 >= 0 and Q2 > 66.6661} = not
    Comprehensive; Comprehensive is the either rule. ``load_two_score_indicators()``:
    elementary/middle schools with all eight indicators, Q = (mean growth score - c,
    weighted mean of the other six - c), weights (0.5, 0.5), cutoff 0.
X = school characteristics from the same year's Report Card Public Data Set that are not
    test-based: log enrollment; % low income, EL, IEP, homeless and race shares; student
    mobility rate; Title I program; school type, grades served, charter, Chicago Public
    Schools; district type, log enrollment, % low income, % EL and number of schools; ROE
    region (first two RCDTS digits), grade band and year indicators. No indicator scores,
    attendance or graduation rates.
Y = the next year's All-Students index score in the same grade band (2023 -> 2024,
    2024 -> 2025). Illinois replaced the index with "Profiles of Performance" in 2026, so
    2025 has no outcome and serves for the rule check and as the 2024 outcome.

Index and indicator scores come from ISBE's public Report Card Service (see download.py).
It has them for 2023-2025 only; 2018, 2019 and 2022 have no published school-level index
and are not used.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample
from experiments.datasets.fl_school_grades._spreadsheet import read, sheet_names
from experiments.datasets.il_school_designations._xlsx import read_columns
from experiments.datasets.il_school_designations.download import (
    DESIGNATIONS, RAW, RCC_DIR, RCC_STATE_DIR, REPORT_CARD, rcdts)

HERE = Path(__file__).parent
PROCESSED = HERE / "data" / "processed"
CACHE = PROCESSED / "il_school_designations_panel.csv.gz"

YEARS = (2023, 2024, 2025)
MAIN_YEARS = (2023, 2024)               # next year's index observed
BANDS = ("ES", "HS")
HALF = 0.00005                          # half the 4-decimal grid step of the index
# Thresholds (Comprehensive T5, Exemplary T10) as printed, to 0.01, in the Descriptive
# Statistics PDFs. The adapter uses the full-precision values of the state
# SummativeSchoolDistribution report (``state_thresholds``), which round to these.
PUBLISHED_THRESHOLDS = {
    (2023, "ES"): (35.39, 81.45), (2023, "HS"): (11.99, 91.24),
    (2024, "ES"): (40.00, 84.04), (2024, "HS"): (9.34, 91.26),
    (2025, "ES"): (38.71, 81.99), (2025, "HS"): (15.03, 93.02),
}
INDICATORS = {
    "ES": {"ELAProficiency": 7.5, "MathProficiency": 7.5, "ELAGrowth": 25.0,
           "MathGrowth": 25.0, "ScienceProficiency": 5.0, "ELPTP": 5.0,
           "ChronicAbsenteeism": 20.0, "ClimateSurvey": 5.0},
    "HS": {"ELAProficiency": 7.5, "MathProficiency": 7.5, "GraduationRate": 50.0,
           "ScienceProficiency": 5.0, "ELPTP": 5.0, "ChronicAbsenteeism": 10.0,
           "ClimateSurvey": 6.67, "Grade9OnTrack": 8.33},
}
SQSS = {"ChronicAbsenteeism", "ClimateSurvey", "Grade9OnTrack"}
ALL_INDICATORS = ["ELAProficiency", "MathProficiency", "ELAGrowth", "MathGrowth",
                  "GraduationRate", "ScienceProficiency", "ELPTP", "ChronicAbsenteeism",
                  "ClimateSurvey", "Grade9OnTrack"]
SHORT = {"ELAProficiency": "ela_prof", "MathProficiency": "math_prof",
         "ELAGrowth": "ela_growth", "MathGrowth": "math_growth",
         "GraduationRate": "grad", "ScienceProficiency": "sci_prof", "ELPTP": "elptp",
         "ChronicAbsenteeism": "chronic", "ClimateSurvey": "climate",
         "Grade9OnTrack": "ontrack"}
# Graduation indicator score = 3.7975 x composite rate - 253.16456, floored at 0 and
# capped at 100: 0 points at a composite rate <= 66.667 %, 100 points at >= 93 %.
GRAD_SLOPE, GRAD_INTERCEPT = 3.7975, 253.16456
GRAD_CUT = GRAD_INTERCEPT / GRAD_SLOPE  # 66.6661 %
COMPREHENSIVE = ("Comprehensive", "Intensive")
_CITATION = ("Illinois State Board of Education, Summative Designation Business Rules "
             "2023-2025, official summative designation lists 2023-2025, Report Card "
             "Service (rcc.isbe.net) SummativeWeightedIndicators, and Report Card Public "
             "Data Sets 2023-2025")


# --------------------------------------------------------------------- raw files

def _num(v) -> float:
    try:
        return float(str(v).strip().replace(",", ""))
    except (TypeError, ValueError):
        return np.nan


def _parse_rcc(path: Path) -> tuple[list[dict], list[dict]]:
    """Indicator rows and "Overall" rows of one Report Card Service file. Two layouts:
    <ReportData> (header + rows; has the Overall rows with the four-decimal index and
    the designation) and a bare <Record> layout (no Overall rows)."""
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    sid = path.stem
    recs = []
    if "<ReportData" in text[:800]:
        head = text.split("</Header>")[0]
        ids = dict(re.findall(r'<Column ndx="(\d+)" id="(\w+)"', head))
        for row in re.findall(r"<Row>(.*?)</Row>", text, re.S):
            cells = re.findall(r'<Column ndx="(\d+)"[^>]*?(?:/>|>([^<]*)</Column>)', row)
            recs.append({ids[n]: v for n, v in cells if n in ids})
    else:
        for row in re.findall(r"<Record>(.*?)</Record>", text, re.S):
            recs.append({t: v for t, v in re.findall(r"<(\w+)>([^<]*)</\1>", row)})
    ind, overall = [], []
    for r in recs:
        base = {"rcdts": sid, "year": int(_num(r.get("Year"))),
                "band": (r.get("GradeBandCode") or "").strip()}
        code = (r.get("IndicatorCode") or "").strip()
        if code == "Overall":
            overall.append({**base, "index4": _num(r.get("OverallIndicatorScore")),
                            "designation_rcc": (r.get("SummativeDesignation") or "").strip()})
        else:
            ind.append({**base, "group": (r.get("SubgroupCode") or "").strip(),
                        "indicator": code, "score": _num(r.get("RawScore")),
                        "weighted": _num(r.get("WeightedScore")),
                        "weight": _num(r.get("IndicatorWeight"))})
    return ind, overall


def _scores() -> pd.DataFrame:
    """One row per school x year x band with an All-Students index: the index (four
    decimals where the file has it), the All-Students indicator scores and effective
    weights, and the student-group index scores."""
    files = sorted(RCC_DIR.glob("*.xml"))
    if not files:
        raise FileNotFoundError(
            f"{RCC_DIR} is empty. Run python -m experiments.datasets.il_school_designations.download")
    ind, ov = [], []
    for f in files:
        a, b = _parse_rcc(f)
        ind += a
        ov += b
    ind = pd.DataFrame(ind)
    ov = pd.DataFrame(ov, columns=["rcdts", "year", "band", "index4", "designation_rcc"])
    key = ["rcdts", "year", "band"]
    idx = ind[ind.indicator == "All"]
    alls = idx[idx.group == "All"].set_index(key)["score"].rename("index2")
    groups = idx[(idx.group != "All") & idx.score.notna()]
    g = groups.groupby(key)
    gstat = pd.DataFrame({"min_group_index": g.score.min(), "n_groups": g.size()})
    glist = groups.assign(pair=groups.group + "=" + groups.score.map("{:.2f}".format)) \
        .groupby(key).pair.apply(";".join).rename("group_indices")
    a = ind[(ind.group == "All") & (ind.indicator != "All")]
    sc = a.pivot_table(index=key, columns="indicator", values="score", aggfunc="first")
    wt = a.pivot_table(index=key, columns="indicator", values="weight", aggfunc="first")
    sc = sc.reindex(columns=ALL_INDICATORS).rename(columns=lambda c: f"s_{SHORT[c]}")
    wt = wt.reindex(columns=ALL_INDICATORS).rename(columns=lambda c: f"w_{SHORT[c]}")
    out = pd.concat([alls, gstat, glist, sc, wt], axis=1).reset_index()
    out = out.merge(ov.drop_duplicates(key), on=key, how="left")
    out = out[out.index2.notna() | out.index4.notna()]
    out["index"] = out.index4.where(out.index4.notna(), out.index2)
    out["index_precision"] = np.where(out.index4.notna(), 4, 2)
    out["n_groups"] = out.n_groups.fillna(0)
    return out.reset_index(drop=True)


def effective_weights(p: pd.DataFrame) -> pd.DataFrame:
    """Process 4 weights (in %): each indicator's nominal weight, with the weight of the
    missing indicators of its category (academic 75, SQSS 25) spread over the present
    ones in proportion to their weights, R_mn = W_mn x missing / remaining."""
    out = pd.DataFrame(np.nan, index=p.index, columns=[f"e_{SHORT[c]}" for c in ALL_INDICATORS])
    for band, w in INDICATORS.items():
        m = (p.band == band).to_numpy()
        for sqss in (False, True):
            names = [c for c in w if (c in SQSS) == sqss]
            present = p.loc[m, [f"s_{SHORT[c]}" for c in names]].notna().to_numpy()
            nominal = np.array([w[c] for c in names])
            scale = nominal.sum() / (present * nominal).sum(axis=1)
            for j, c in enumerate(names):
                out.loc[m, f"e_{SHORT[c]}"] = np.where(present[:, j], nominal[j] * scale, np.nan)
    return out


def _designations() -> pd.DataFrame:
    """The official lists: school x year x band -> designation and reason."""
    rows = []
    for y, rel in DESIGNATIONS.items():
        path = RAW / rel
        for i, sheet in enumerate(sheet_names(path)):
            grid = read(path, i)
            h = next(k for k, r in enumerate(grid)
                     if r and str(r[0] or "").strip().startswith("RCDTS"))
            for r in grid[h + 1:]:
                if not r or not (sid := rcdts(r[0])):
                    continue
                r = list(r) + [None] * 6
                rows.append({"rcdts": sid, "year": y,
                             "band": str(r[3] or "").strip(),
                             "designation_published": str(r[4] or "").strip(),
                             "reason_published": " ".join(str(r[5] or "").split()),
                             "list_sheet": sheet, "district_name": str(r[1] or "").strip(),
                             "school_name": " ".join(str(r[2] or "").split())})
    d = pd.DataFrame(rows)
    return d.drop_duplicates(["rcdts", "year", "band"]).reset_index(drop=True)


_RC = {
    "rcdts": ["RCDTS"], "type": ["Type", "Level"], "school_type": ["School Type"],
    "district_type": ["District Type"], "grades": ["Grades Served"], "county": ["County"],
    "title1": ["Title 1 Status", "Title I Status"],
    "rc_designation": ["Summative Designation"],
    "enroll": ["# Student Enrollment"],
    "pct_white": ["% Student Enrollment - White"],
    "pct_black": ["% Student Enrollment - Black or African American"],
    "pct_hispanic": ["% Student Enrollment - Hispanic or Latino"],
    "pct_asian": ["% Student Enrollment - Asian"],
    "pct_iep": ["% Student Enrollment - IEP"],
    "pct_el": ["% Student Enrollment - EL"],
    "pct_low_income": ["% Student Enrollment - Low Income"],
    "pct_homeless": ["% Student Enrollment - Homeless"],
    "mobility": ["Student Mobility Rate"],
    "grad4": ["High School 4-Year Graduation Rate - Total"],
    "grad5": ["High School 5-Year Graduation Rate - Total"],
    "grad6": ["High School 6-Year Graduation Rate - Total"],
}
_RC_NUM = ["enroll", "pct_white", "pct_black", "pct_hispanic", "pct_asian", "pct_iep",
           "pct_el", "pct_low_income", "pct_homeless", "mobility", "grad4", "grad5", "grad6"]


def _report_card(year: int) -> pd.DataFrame:
    """School rows of the year's "General" sheet with the district's enrollment, % low
    income and % EL attached."""
    path = RAW / REPORT_CARD[year]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run python -m experiments.datasets.il_school_designations.download")
    rc = pd.DataFrame(read_columns(path, "General", _RC))
    for c in _RC_NUM:
        rc[c] = rc[c].map(_num)
    for c in ["type", "school_type", "district_type", "grades", "county", "title1",
              "rc_designation"]:
        rc[c] = rc[c].fillna("").astype(str).str.strip()
    rc["rcdts"] = rc.rcdts.map(rcdts)
    rc = rc[rc.rcdts != ""]
    rc["district_id"] = rc.rcdts.str[:11]
    dist = rc[rc.type == "District"][["district_id", "enroll", "pct_low_income", "pct_el"]]
    dist = dist.drop_duplicates("district_id").rename(columns={
        "enroll": "district_enroll", "pct_low_income": "district_pct_low_income",
        "pct_el": "district_pct_el"})
    sch = rc[rc.type == "School"].drop(columns="type").drop_duplicates("rcdts")
    sch = sch.merge(dist, on="district_id", how="left")
    nsch = sch.groupby("district_id").size().rename("district_n_schools")
    sch = sch.merge(nsch, on="district_id", how="left")
    sch["year"] = year
    return sch


def _designations_2022() -> pd.Series:
    """rcdts -> 2022 summative designation, from the 2022 report card (one per school; a
    dually designated school shows one of its two)."""
    path = RAW / REPORT_CARD[2022]
    rows = read_columns(path, "General", {"rcdts": ["RCDTS"], "type": ["Type", "Level"],
                                          "d": ["Summative Designation"]})
    out = {rcdts(r["rcdts"]): str(r["d"] or "").strip() for r in rows if r["type"] == "School"}
    return pd.Series(out)


_GRADE = {"PK": -1, "K": 0, "KG": 0}


def _grade(g: str) -> float:
    g = g.strip().upper()
    return float(_GRADE[g]) if g in _GRADE else (float(g) if g.isdigit() else np.nan)


# ---------------------------------------------------------------------------- rule

def state_thresholds() -> pd.DataFrame:
    """ISBE's thresholds at full precision, from the state SummativeSchoolDistribution
    report (columns ComprehensiveTargetedThreshold, ExemplaryCommendableThreshold of the
    All-Students index row), with the PDF values for comparison."""
    rows = []
    for f in sorted(RCC_STATE_DIR.glob("SummativeSchoolDistribution_*.xml")):
        text = f.read_text(encoding="utf-8-sig", errors="replace")
        if "<ReportData" in text[:800]:
            head = text.split("</Header>")[0]
            ids = dict(re.findall(r'<Column ndx="(\d+)" id="(\w+)"', head))
            recs = [{ids[n]: v for n, v in re.findall(
                r'<Column ndx="(\d+)"[^>]*?(?:/>|>([^<]*)</Column>)', row) if n in ids}
                for row in re.findall(r"<Row>(.*?)</Row>", text, re.S)]
        else:
            recs = [{t: v for t, v in re.findall(r"<(\w+)>([^<]*)</\1>", row)}
                    for row in re.findall(r"<Record>(.*?)</Record>", text, re.S)]
        for r in recs:
            t5, t10 = _num(r.get("ComprehensiveTargetedThreshold")), _num(
                r.get("ExemplaryCommendableThreshold"))
            if np.isfinite(t5) or np.isfinite(t10):
                y, b = int(_num(r.get("Year"))), (r.get("GradeBandCode") or "").strip()
                pdf = PUBLISHED_THRESHOLDS.get((y, b), (np.nan, np.nan))
                rows.append({"year": y, "band": b, "t5_full": t5, "t10_full": t10,
                             "t5_pdf": pdf[0], "t10_pdf": pdf[1]})
    if not rows:
        raise FileNotFoundError(
            f"{RCC_STATE_DIR} is empty. Run python -m experiments.datasets.il_school_designations.download")
    return pd.DataFrame(rows).drop_duplicates(["year", "band"]).reset_index(drop=True)


def thresholds(p: pd.DataFrame) -> pd.DataFrame:
    """Process 5 thresholds recomputed from the published All-Students index scores:
    rank in descending order; T10 = score at rank round(0.1 n), T5 = score at rank
    round(0.95 n) + 1 (round half up), next to ISBE's full-precision thresholds and the
    ranks at which those fall in the public list. The public list lacks the elementary
    band of dually designated schools (see README), so the elementary ranks differ."""
    st = state_thresholds().set_index(["year", "band"])
    rows = []
    for (y, b), g in p[p["index"].notna()].groupby(["year", "band"]):
        s = np.sort(g["index"].to_numpy())[::-1]
        n = len(s)
        r10, r5 = int(np.floor(0.1 * n + 0.5)), int(np.floor(0.95 * n + 0.5)) + 1
        t = st.loc[(y, b)]
        t5, t10 = round(t.t5_full, 4), round(t.t10_full, 4)
        rows.append({"year": y, "band": b, "n_ranked": n, "rank_t10": r10, "rank_t5": r5,
                     "t10_rank_rule": s[r10 - 1], "t5_rank_rule": s[r5 - 1],
                     "t10_full": t.t10_full, "t5_full": t.t5_full,
                     "t10_pdf": t.t10_pdf, "t5_pdf": t.t5_pdf,
                     "n_at_or_above_t10": int((s >= t10 - 1e-9).sum()),
                     "n_at_or_below_t5": int((s <= t5 + 1e-9).sum())})
    return pd.DataFrame(rows)


def state_rule(p: pd.DataFrame) -> pd.DataFrame:
    """The Process 5 designation from the published scores: comp_index, comp_grad,
    targeted, exemplary and the resulting designation_rule."""
    out = pd.DataFrame(index=p.index)
    # The index has four decimals; t5 and t10 are ISBE's thresholds rounded to four.
    comp_i = p["index"] <= p.t5 + 1e-9
    comp_g = (p.band == "HS") & (p.s_grad == 0) & ~comp_i
    # A student group at or below T5. Group index scores are published to two decimals,
    # so they are compared with the full-precision threshold.
    low_group = p.min_group_index.notna() & (p.min_group_index <= p.t5_full)
    targeted = ~comp_i & ~comp_g & low_group
    exemp = ~comp_i & ~comp_g & ~targeted & (p["index"] >= p.t10 - 1e-9)
    out["group_tie"] = (p.min_group_index.round(2) == p.t5_full.round(2)).astype(float)
    out["comp_index"] = comp_i.astype(float)
    out["comp_grad"] = comp_g.astype(float)
    out["targeted_rule"] = targeted.astype(float)
    out["exemplary_rule"] = exemp.astype(float)
    out["designation_rule"] = np.select(
        [comp_i, comp_g, targeted, exemp],
        ["Comprehensive", "Comprehensive", "Targeted", "Exemplary"], "Commendable")
    return out


def _explained(p: pd.DataFrame) -> pd.Series:
    """Published designation consistent with the rule designation once the funding-table
    steps are allowed: Intensive for a rule-Comprehensive school, Comprehensive "From
    Targeted" for a rule-Targeted school."""
    pub, rule, why = p.designation_published, p.designation_rule, p.reason_published
    ok = pub == rule
    ok |= (pub == "Intensive") & (rule == "Comprehensive")
    ok |= (pub == "Comprehensive") & (rule == "Targeted") & why.str.contains("From Targeted")
    return ok


# --------------------------------------------------------------------------- panel

def build() -> pd.DataFrame:
    """Every school x year x band with an index score in 2023-2025: scores, thresholds,
    rule and published designations, report-card covariates, next year's index (Y) and
    last year's index."""
    if not CACHE.exists():
        df = _build()
        PROCESSED.mkdir(parents=True, exist_ok=True)
        tmp = CACHE.with_name(CACHE.name + ".tmp")
        df.to_csv(tmp, index=False, compression="gzip")
        tmp.replace(CACHE)
    p = pd.read_csv(CACHE, keep_default_na=False, na_values=[""], low_memory=False,
                    dtype={"rcdts": str, "district_id": str, "region": str})
    for c in ["band", "designation_published", "reason_published", "designation_rule",
              "designation_rcc", "group_indices", "school_type", "district_type",
              "grades", "county", "title1", "title1_program", "rc_designation",
              "list_sheet", "district_name", "school_name", "grad_rate_source",
              "y_designation", "designation_prev", "school_kind", "district_kind"]:
        p[c] = p[c].fillna("").astype(str)
    return p


def _build() -> pd.DataFrame:
    p = _scores()
    p = p[p.year.isin(YEARS) & p.band.isin(BANDS)].reset_index(drop=True)
    e = effective_weights(p)
    p = pd.concat([p, e], axis=1)
    S = p[[f"s_{SHORT[c]}" for c in ALL_INDICATORS]].to_numpy(float)
    p["index_components"] = np.nansum(S * e.to_numpy(float), axis=1) / 100
    p["n_indicators"] = np.isfinite(S).sum(axis=1)
    p["n_sqss"] = p[["s_chronic", "s_climate", "s_ontrack"]].notna().sum(axis=1)
    st = state_thresholds()[["year", "band", "t5_full", "t10_full"]]
    p = p.merge(st, on=["year", "band"], how="left")
    p["t5"], p["t10"] = p.t5_full.round(4), p.t10_full.round(4)
    p = pd.concat([p, state_rule(p)], axis=1)
    p = p.merge(_designations(), on=["rcdts", "year", "band"], how="left")
    for c in ["designation_published", "reason_published", "list_sheet"]:
        p[c] = p[c].fillna("")
    p["from_targeted"] = p.reason_published.str.contains("From Targeted").astype(float)
    p["rule_agrees"] = _explained(p).astype(float)
    p["comprehensive_published"] = p.designation_published.isin(COMPREHENSIVE).astype(float)
    # Composite graduation rate: inverted from the indicator score where it is not
    # censored (0 < score < 100), else from the report card's 4-, 5- and 6-year rates.
    rc = pd.concat([_report_card(y) for y in YEARS], ignore_index=True)
    p = p.merge(rc, on=["rcdts", "year"], how="left")
    p["grad_rc"] = 0.6 * p.grad4 + 0.3 * p.grad5 + 0.1 * p.grad6
    # The report card's rate is for the responsible school, the indicator's for the
    # accountable school, so the two can differ; where the score is censored, the report
    # card's rate is clipped to the side of the censoring point that the score implies.
    inv = (p.s_grad + GRAD_INTERCEPT) / GRAD_SLOPE
    unc = (p.s_grad > 0) & (p.s_grad < 100)
    top = (100 + GRAD_INTERCEPT) / GRAD_SLOPE                    # 92.9993 %
    rc_side = np.where(p.s_grad == 0, np.fmin(p.grad_rc, GRAD_CUT),
                       np.where(p.s_grad == 100, np.fmax(p.grad_rc, top), np.nan))
    p["grad_rate"] = inv.where(unc, rc_side)
    clipped = ((p.s_grad == 0) & (p.grad_rc > GRAD_CUT)) | ((p.s_grad == 100) & (p.grad_rc < top))
    p["grad_rate_source"] = np.select(
        [unc, p.grad_rate.notna() & clipped, p.grad_rate.notna()],
        ["score", "report_card_clipped", "report_card"], "")
    t1 = p.title1.fillna("").str.lower()
    p["title1_program"] = np.select(
        [t1.str.contains("schoolwide"), t1.str.contains("targeted")],
        ["schoolwide", "targeted"], np.where(t1 == "", "", "not served"))
    p["region"] = p.rcdts.str[:2]
    p["charter"] = (p.rcdts.str[-1] == "C").astype(float)
    p["cps"] = (p.rcdts.str[:11] == "15016299025").astype(float)
    span = p.grades.fillna("").str.replace(" ", "").str.split("-", n=1, expand=True) \
        .reindex(columns=[0, 1]).fillna("")
    p["low_grade"] = span[0].map(_grade)
    p["high_grade"] = span[1].map(_grade).where(span[1] != "", span[0].map(_grade))
    # The type labels change by year (2023: "ELEMENTARY", "MIDDLE SCHL", "HIGH SCHOOL",
    # "CHARTER SCH", "UNIT"; 2024: "Unit Distrcts"; 2025: "Unit District"). 2023 charters
    # get the type their grades imply.
    st = p.school_type.fillna("").str.lower()
    by_grades = np.where(p.low_grade >= 9, "high",
                         np.where((p.high_grade >= 9) & (p.low_grade >= 6), "high",
                                  np.where((p.low_grade >= 5) & (p.high_grade <= 9), "middle",
                                           "elementary")))
    p["school_kind"] = np.select(
        [st.str.contains("elem"), st.str.contains("middle"), st.str.contains("high"),
         st.str.contains("special")], ["elementary", "middle", "high", "special"], by_grades)
    dt = p.district_type.fillna("").str.lower()
    p["district_kind"] = np.select(
        [dt.str.contains("elem"), dt.str.contains("high"), dt.str.contains("unit")],
        ["elementary", "high", "unit"], "other")
    # Next year's and last year's index, same school and band.
    key = ["rcdts", "band"]
    nxt = p[key + ["year", "index", "designation_published"]].rename(columns={
        "index": "y_index", "designation_published": "y_designation"})
    nxt["year"] = nxt.year - 1
    p = p.merge(nxt, on=key + ["year"], how="left")
    lag = p[key + ["year", "index"]].rename(columns={"index": "q_lag"})
    lag["year"] = lag.year + 1
    p = p.merge(lag, on=key + ["year"], how="left")
    p["y_year"] = p.year + 1
    # Previous year's published designation, same band (2023: the school's single 2022
    # designation shown on the 2022 report card). Support lasts four years from
    # identification, so a school designated Comprehensive or Targeted the year before is
    # already in support whatever this year's designation is.
    prev = p[key + ["year", "designation_published"]].rename(
        columns={"designation_published": "designation_prev"})
    prev["year"] = prev.year + 1
    p = p.merge(prev, on=key + ["year"], how="left")
    d22 = _designations_2022()
    m = p.year == 2023
    p.loc[m, "designation_prev"] = p.loc[m, "rcdts"].map(d22)
    p["designation_prev"] = p.designation_prev.fillna("")
    p["comprehensive_prev"] = p.designation_prev.isin(COMPREHENSIVE).astype(float)
    return p.sort_values(["year", "band", "rcdts"]).reset_index(drop=True)


# -------------------------------------------------------------------------- checks

def rule_check() -> pd.DataFrame:
    """Per year and band: thresholds recomputed against the published ones, and the
    published designation against the Process 5 rule computed from the published index
    scores (exact, and allowing the funding-table steps)."""
    p = build()
    th = thresholds(p).set_index(["year", "band"])
    lists = _designations()
    rows = []
    for (y, b), g in p.groupby(["year", "band"]):
        t = th.loc[(y, b)]
        listed = g.designation_published != ""
        n_list = int(((lists.year == y) & (lists.band == b)).sum())
        rows.append({
            "year": y, "band": b, "on_list": n_list, "with_index": len(g),
            "listed_without_index": n_list - int(listed.sum()),
            "t5_full": round(t.t5_full, 6), "t5_pdf": t.t5_pdf,
            "t5_rank_rule": t.t5_rank_rule, "n_at_or_below_t5": t.n_at_or_below_t5,
            "t10_full": round(t.t10_full, 6), "t10_pdf": t.t10_pdf,
            "t10_rank_rule": t.t10_rank_rule, "n_at_or_above_t10": t.n_at_or_above_t10,
            "group_ties": int(g.group_tie.sum()),
            "max_abs_index_minus_components": float((g["index"] - g.index_components).abs().max()),
            "comp_index": int(g.comp_index.sum()), "comp_grad": int(g.comp_grad.sum()),
            "targeted": int(g.targeted_rule.sum()), "exemplary": int(g.exemplary_rule.sum()),
            "exact_agree": float((g.designation_rule == g.designation_published)[listed].mean()),
            "agree_with_funding_steps": float(g.rule_agrees[listed].mean()),
            "comprehensive_agree": float(((g.comp_index + g.comp_grad > 0)
                                          == (g.comprehensive_published == 1))[listed].mean()),
            "rcc_designation_agree": float((g.designation_rcc == g.designation_published)[
                listed & (g.designation_rcc != "")].mean()),
        })
    return pd.DataFrame(rows)


def exceptions() -> pd.DataFrame:
    """Listed school-years whose published designation the rule does not explain."""
    p = build()
    m = (p.designation_published != "") & (p.rule_agrees == 0)
    return p.loc[m, ["year", "band", "rcdts", "school_name", "index", "t5", "t10",
                     "min_group_index", "s_grad", "designation_rule",
                     "designation_published", "reason_published"]]


# ------------------------------------------------------------------------- samples

def _design(df: pd.DataFrame) -> tuple[np.ndarray, list]:
    X = pd.DataFrame(index=df.index)

    def fill(name, v):
        X[name] = v.fillna(v.median())
        if v.isna().any():
            X[f"{name}_missing"] = v.isna().astype(float)

    fill("log_enrollment", np.log(df.enroll.where(df.enroll > 0)))
    # The report card suppresses a group's share when the group has fewer than 10
    # students; such a share is set to the midpoint, 4.5 students / enrollment.
    small = (100 * 4.5 / df.enroll).clip(upper=100)
    shares = {c: df[c].fillna(small) for c in
              ["pct_low_income", "pct_el", "pct_iep", "pct_homeless", "pct_white",
               "pct_black", "pct_hispanic", "pct_asian"]}
    for c in ["pct_low_income", "pct_el", "pct_iep", "pct_homeless", "pct_black",
              "pct_hispanic", "pct_asian"]:
        fill(c, shares[c])
    fill("pct_other_race", (100 - shares["pct_white"] - shares["pct_black"]
                            - shares["pct_hispanic"] - shares["pct_asian"]).clip(lower=0))
    fill("mobility", df.mobility)
    fill("district_log_enrollment", np.log(df.district_enroll.where(df.district_enroll > 0)))
    fill("district_pct_low_income", df.district_pct_low_income)
    fill("district_pct_el", df.district_pct_el)
    fill("district_log_n_schools", np.log(df.district_n_schools))
    fill("low_grade", df.low_grade)
    fill("high_grade", df.high_grade)
    X["charter"] = df.charter
    X["chicago_ps"] = df.cps
    X["hs_band"] = (df.band == "HS").astype(float)
    for t in ["middle", "high", "special"]:
        X[f"school_{t}"] = (df.school_kind == t).astype(float)
    for t in ["elementary", "high", "other"]:
        X[f"district_{t}"] = (df.district_kind == t).astype(float)
    # Title I labels differ by year (2023: "Schoolwide Title I Program", "Targeted
    # Assistance Title I Pr..."; 2024-2025: "Title 1 Schoolwide", "Title 1 Targeted").
    X["title1_schoolwide"] = (df.title1_program == "schoolwide").astype(float)
    X["title1_targeted"] = (df.title1_program == "targeted").astype(float)
    for y in sorted(df.year.unique())[1:]:
        X[f"year_{y}"] = (df.year == y).astype(float)
    counts = df.region.value_counts()
    base = counts.index[0]
    for r in sorted(counts.index):
        if r != base:
            X[f"region_{r}"] = (df.region == r).astype(float)
    X = X.loc[:, X.std() > 0]
    return X.to_numpy(float), list(X.columns)


def _frame(years=MAIN_YEARS, bands=BANDS, y_col: str = "y_index") -> pd.DataFrame:
    """Listed school-years of the main years with the report-card row and Y observed."""
    p = build()
    m = (p.year.isin(years) & p.band.isin(bands) & (p.designation_published != "")
         & p.enroll.notna() & p[y_col].notna())
    return p[m].reset_index(drop=True)


def _extras(df: pd.DataFrame) -> dict:
    group = (df.year.astype(str) + "-" + df.band).to_numpy()
    return {
        "group": group, "year": df.year.to_numpy(int), "band": df.band.to_numpy(str),
        "rcdts": df.rcdts.to_numpy(str), "district_id": df.district_id.to_numpy(str),
        "region": df.region.to_numpy(str), "school_name": df.school_name.to_numpy(str),
        "index": df["index"].to_numpy(float), "index_precision": df.index_precision.to_numpy(int),
        "t5": df.t5.to_numpy(float), "t10": df.t10.to_numpy(float),
        "min_group_index": df.min_group_index.to_numpy(float),
        "n_groups": df.n_groups.to_numpy(float),
        "grad_score": df.s_grad.to_numpy(float), "grad_rate": df.grad_rate.to_numpy(float),
        "indicator_scores": df[[f"s_{SHORT[c]}" for c in ALL_INDICATORS]].to_numpy(float),
        "indicator_weights": df[[f"w_{SHORT[c]}" for c in ALL_INDICATORS]].to_numpy(float),
        "indicator_names": tuple(SHORT[c] for c in ALL_INDICATORS),
        "designation_published": df.designation_published.to_numpy(str),
        "reason_published": df.reason_published.to_numpy(str),
        "designation_rule": df.designation_rule.to_numpy(str),
        "title1": df.title1.to_numpy(str),
        "y_year": df.y_year.to_numpy(int), "y_designation": df.y_designation.fillna("").to_numpy(str),
        "q_lag": df.q_lag.to_numpy(float),
        "designation_prev": df.designation_prev.to_numpy(str),
        "comprehensive_prev": df.comprehensive_prev.to_numpy(float),
        "t5_full": df.t5_full.to_numpy(float), "t10_full": df.t10_full.to_numpy(float),
    }


def _comp_frame(bands=BANDS) -> tuple[pd.DataFrame, dict]:
    df = _frame(bands=bands)
    drop = {"comp_grad": int((df.comp_grad == 1).sum()),
            "from_targeted": int((df.from_targeted == 1).sum())}
    df = df[(df.comp_grad == 0) & (df.from_targeted == 0)].reset_index(drop=True)
    return df, drop


def _comprehensive(bands=BANDS, name: str = "il_school_designations",
                   not_in_support: bool = False) -> RDDSample:
    df, drop = _comp_frame(bands)
    if not_in_support:
        drop["comprehensive_prev"] = int((df.comprehensive_prev == 1).sum())
        df = df[df.comprehensive_prev == 0].reset_index(drop=True)
    X, names = _design(df)
    cut = (df.t5 + HALF).to_numpy(float)
    ex = _extras(df)
    ex.update({
        "cutoff_index": cut, "group_cutoffs": dict(zip(ex["group"], cut)),
        "eligible_published": (df.comprehensive_published == 0).to_numpy(float),
        "excluded": drop, "boundary": "comprehensive",
        "treated_side": "D = 1 = not Comprehensive (index above T5); Comprehensive "
                        "(IL-EMPOWER support and funding) is the untreated side Q < 0",
    })
    which = {("ES", "HS"): "ES and HS bands", ("ES",): "elementary/middle band",
             ("HS",): "high school band"}[tuple(bands)]
    return RDDSample(
        Q=(df["index"] - cut).to_numpy(float), X=X, Y=df.y_index.to_numpy(float),
        threshold=0.0, name=name, feature_names=names,
        description=(f"Illinois ESSA summative designations, 2023-2024, {which}. "
                     "Q = All-Students index - (T5 + 0.00005), T5 the year-and-band "
                     "bottom-5% threshold; treatment = 1{Q >= 0} (not Comprehensive); "
                     "Y = next year's index score."),
        citation=_CITATION, treatment_rule=lambda Q: (Q >= 0).astype(int), extras=ex,
    )


def load() -> RDDSample:
    """Comprehensive boundary, ES and HS pooled: Q = index - (T5 + 0.00005) by year and
    band; D = 1{Q >= 0} = not Comprehensive. Excludes HS designated Comprehensive by the
    graduation rule alone and Targeted schools escalated to Comprehensive."""
    return _comprehensive()


def load_es() -> RDDSample:
    """``load()`` for the elementary/middle band only."""
    return _comprehensive(("ES",), "il_school_designations_es")


def load_hs() -> RDDSample:
    """``load()`` for the high school band only."""
    return _comprehensive(("HS",), "il_school_designations_hs")


def load_not_in_support() -> RDDSample:
    """``load()`` without the school-years designated Comprehensive or Intensive the year
    before (already in a four-year support cycle), so that D = 0 is a new identification."""
    return _comprehensive(name="il_school_designations_not_in_support", not_in_support=True)


def load_exemplary() -> RDDSample:
    """Exemplary/Commendable boundary: listed school-years that the rule designates
    neither Comprehensive nor Targeted; Q = index - T10, D = 1{Q >= 0} = Exemplary."""
    df = _frame()
    df = df[df.designation_rule.isin(["Exemplary", "Commendable"])
            & (df.from_targeted == 0)].reset_index(drop=True)
    X, names = _design(df)
    cut = df.t10.to_numpy(float)
    ex = _extras(df)
    ex.update({"cutoff_index": cut, "group_cutoffs": dict(zip(ex["group"], cut)),
               "eligible_published": (df.designation_published == "Exemplary").to_numpy(float),
               "boundary": "exemplary",
               "treated_side": "D = 1 = Exemplary (index at or above T10)"})
    return RDDSample(
        Q=(df["index"] - cut).to_numpy(float), X=X, Y=df.y_index.to_numpy(float),
        threshold=0.0, name="il_school_designations_exemplary", feature_names=names,
        description=("Illinois ESSA summative designations, 2023-2024, schools with no "
                     "student group at or below T5. Q = index - T10 (top-10% threshold); "
                     "treatment = 1{Q >= 0} (Exemplary); Y = next year's index score."),
        citation=_CITATION, treatment_rule=lambda Q: (Q >= 0).astype(int), extras=ex,
    )


def load_two_score() -> RDDSample:
    """High schools: Q = (index - (T5 + 0.00005), composite graduation rate). D = 1{Q1 >= 0
    and Q2 > 66.6661} = not Comprehensive; Comprehensive is the either rule (index <= T5
    or graduation rate <= 66.67%). Excludes Targeted schools escalated to Comprehensive."""
    df = _frame(bands=("HS",))
    df = df[(df.from_targeted == 0) & df.grad_rate.notna()].reset_index(drop=True)
    X, names = _design(df)
    cut = (df.t5 + HALF).to_numpy(float)
    q = np.column_stack([df["index"] - cut, df.grad_rate])
    cuts = (0.0, GRAD_CUT)
    ex = _extras(df)
    ex.update({
        "cutoffs": cuts, "rule": "both", "rule_comprehensive": "either",
        "cutoff_index": cut, "grad_rate_source": df.grad_rate_source.to_numpy(str),
        "eligible_published": (df.comprehensive_published == 0).to_numpy(float),
        "treated_side": "D = 1 = not Comprehensive: Q1 >= 0 and Q2 > 66.6661 (both); "
                        "Comprehensive is Q1 < 0 or Q2 <= 66.6661 (either)",
    })
    return RDDSample(
        Q=q, X=X, Y=df.y_index.to_numpy(float), threshold=cuts,
        name="il_school_designations_two_score", feature_names=names,
        description=("Illinois ESSA summative designations, high schools 2023-2024. Q = "
                     "(index - (T5 + 0.00005), composite graduation rate %); treatment = "
                     "1{Q1 >= 0 and Q2 > 66.6661} (not Comprehensive); Y = next year's index."),
        citation=_CITATION,
        treatment_rule=lambda Q: ((Q[:, 0] >= 0) & (Q[:, 1] > GRAD_CUT)).astype(int),
        extras=ex,
    )


def _indicators(elptp: bool) -> RDDSample:
    df, drop = _comp_frame(bands=("ES",))
    names_es = list(INDICATORS["ES"])
    want = [c for c in names_es if elptp or c != "ELPTP"]
    have = df[[f"s_{SHORT[c]}" for c in names_es]].notna()
    pattern = np.all([have[f"s_{SHORT[c]}"] == (c in want) for c in names_es], axis=0)
    df = df[pattern].reset_index(drop=True)
    X, names = _design(df)
    e = {c: df[f"e_{SHORT[c]}"].iloc[0] for c in want}      # the same in every row
    growth, others = ("ELAGrowth", "MathGrowth"), [c for c in want if c not in ("ELAGrowth", "MathGrowth")]
    wg = sum(e[c] for c in growth)
    g = sum(e[c] * df[f"s_{SHORT[c]}"] for c in growth) / wg
    r = sum(e[c] * df[f"s_{SHORT[c]}"] for c in others) / sum(e[c] for c in others)
    cut = (df.t5 + HALF).to_numpy(float)
    q = np.column_stack([g - cut, r - cut])
    weights = (round(wg / 100, 6), round(1 - wg / 100, 6))
    wv = np.array(weights)
    comp = 0.01 * (wg * g + (100 - wg) * r)
    ex = _extras(df)
    ex.update({"weights": weights, "cutoff": 0.0, "cutoff_index": cut,
               "effective_weights": e, "growth": g.to_numpy(float), "rest": r.to_numpy(float),
               "index_from_components": comp.to_numpy(float),
               "D_index": (df["index"] - cut >= 0).to_numpy(int),
               "eligible_published": (df.comprehensive_published == 0).to_numpy(float),
               "excluded": drop,
               "treated_side": f"D = 1 = not Comprehensive: {weights[0]} Q1 + {weights[1]} Q2 >= 0"})
    tag = "" if elptp else "_no_elptp"
    what = "all eight indicators" if elptp else "every indicator but ELPtP"
    return RDDSample(
        Q=q, X=X, Y=df.y_index.to_numpy(float), threshold=(0.0, 0.0),
        name=f"il_school_designations_two_score_indicators{tag}", feature_names=names,
        description=(f"Illinois ESSA summative designations, elementary/middle schools with "
                     f"{what}, 2023-2024. Q = (growth score - c, weighted mean of the other "
                     f"indicators - c), c = T5 + 0.00005; treatment = 1{{{weights[0]} Q1 + "
                     f"{weights[1]} Q2 >= 0}} (not Comprehensive); Y = next year's index."),
        citation=_CITATION, treatment_rule=lambda Q: (Q @ wv >= 0).astype(int),
        extras=ex,
    )


def load_two_score_indicators() -> RDDSample:
    """Elementary/middle schools with all eight indicators, so every school has the
    nominal weights: Q = (G - c, R - c), G = mean of the ELA and math growth scores (the
    two largest components, 25% each), R = the weighted mean of the other six indicator
    scores, c = T5 + 0.00005 for the year; weights (0.5, 0.5). index = 0.5 G + 0.5 R up to
    the rounding of the published indicator scores (|error| < 0.005), so D = 1{0.5 Q1 +
    0.5 Q2 >= 0} = not Comprehensive. Same exclusions as ``load()``."""
    return _indicators(elptp=True)


def load_two_score_indicators_no_elptp() -> RDDSample:
    """As ``load_two_score_indicators()`` for elementary/middle schools with every
    indicator but ELPtP (fewer than 20 English learners); its 5% academic weight goes to
    the other academic indicators, so the weights are (0.5357, 0.4643)."""
    return _indicators(elptp=False)
