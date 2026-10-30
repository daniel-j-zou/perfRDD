"""Texas campus accountability, 2013-2017 (TEA): Met Standard or Improvement Required
from targets on four performance indexes.

From 2013 to 2017 TEA rated each campus on four indexes, each a score from 0 to 100 equal
to points earned as a percent of maximum points (Accountability Manuals 2013-2017):
Index 1 Student Achievement (% of STAAR tests at the satisfactory standard), Index 2
Student Progress (weighted STAAR growth), Index 3 Closing Performance Gaps (weighted
performance of the economically disadvantaged and lowest-performing racial/ethnic
groups) and Index 4 Postsecondary Readiness (for elementary and middle schools the STAAR
component only, % of students at the postsecondary standard; not evaluated for them in
2013). Each index score is the ratio rounded half up to a whole number (the Index 4
STAAR component is published to one decimal first); a campus meets an index when its
rounded score is at least the target. The rating:

* 2013 and 2014: Met Standard if the campus meets the target on every index for which it
  has data; Improvement Required (IR) otherwise.
* 2015, 2016 and 2017: Met Standard if it meets "Index 1 OR Index 2 AND Index 3 AND
  Index 4" (each counted when the campus has data for it).

Targets for elementary / middle schools (non-AEA; each campus summary prints its own):

    year  Index 1  Index 2   Index 3   Index 4 (STAAR only)
    2013  50       30 / 29   55        not evaluated
    2014  55       33 / 28   28 / 27   12 / 13
    2015  60       30 / 28   28 / 27   12 / 13
    2017  60       32 / 30   28 / 26   12 / 13

Margins m_k = (unrounded index k) - (target_k - 0.5), so "meets index k" is exactly
m_k >= 0 (Index 1-3: 100 * points / maximum points; Index 4: the one-decimal STAAR
component). M12 = min(m1, m2) in 2013-2014 and max(m1, m2) from 2015. IR <=> any required
margin < 0.

Q   ``load()``: min(M12, m3, m4), D = 1{Q < 0} = IR (treated side: below).
    ``load_two_score()``: (M12, min(m3, m4)) by default, which reproduces the rule exactly;
    ``pair="12_3"`` gives (M12, m3) and ``"1_3"`` gives (m1, m3). D = 1{Q1 < 0 or Q2 < 0}.
    ``load_three_score()``: (M12, m3, m4), 2014 on.
X   campus characteristics from the summary's "Campus Demographics" (PEIMS enrollment
    data, not test results): % economically disadvantaged, % English language learners,
    mobility rate, log enrollment, lowest and highest grade served, middle school;
    district log enrollment and % economically disadvantaged (aggregated over the
    district's campus summaries); charter; ESC region and year indicators.
Y   the next rated year's Index 1, unrounded (``load_gain()``: minus this year's). Static
    campus summaries exist for 2013, 2014, 2015 and 2017 only (2016 is form-only), so the
    default sample is the 2013 and 2014 ratings with Y from 2014 and 2015;
    ``years=(2013, 2014, 2015)`` adds 2015 with Y = 2017 Index 1 (``extras["y_year"]``).
    ``load_2017()``: 2017 ratings with Y = the 2018 A-F Student Achievement domain score
    (from the tx_school_ratings dataset).

Sample: non-AEA elementary and middle schools with their own index data (not paired),
rated Met Standard or IR in TEA's final list, with Index 1 and 3 (and Index 4 from 2014)
and Y observed. Index 2 may be missing (then M12 = m1).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample
from experiments.datasets.tx_index_2013.download import RAW, SUMMARIES, SUMMARY_YEARS, YEARS

HERE = Path(__file__).parent
PROCESSED = HERE / "data" / "processed"
CACHE = PROCESSED / "tx_index_2013_panel.csv.gz"

OR_YEARS = (2015, 2016, 2017)            # "Index 1 OR Index 2 AND Index 3 AND Index 4"
MAIN_YEARS = (2013, 2014)                # next-year Index 1 observed
NEXT = {2013: 2014, 2014: 2015, 2015: 2017}   # next year with campus summaries
INDEX1_TARGET = {2013: 50, 2014: 55, 2015: 60, 2016: 60, 2017: 60}      # non-AEA
MAIN_TYPES = ("Elementary", "Middle School")
EPS = 1e-9
_GRADE = {"EE": -2, "PK": -1, "KG": 0, "K": 0}
_RATINGS = {"MET STANDARD": "Met Standard", "IMPROVEMENT REQUIRED": "Improvement Required",
            "MET ALTERNATIVE STANDARD": "Met Alternative Standard", "NOT RATED": "Not Rated",
            "DATA INTEGRITY ISSUES": "Not Rated: Data Integrity Issues"}
INDEX_NAMES = {"Student Achievement": 1, "Student Progress": 2,
               "Closing Performance Gaps": 3, "Postsecondary Readiness": 4}
_TXT = ["campus", "district", "name", "district_name", "rating", "rating_final", "note",
        "rating_change", "binding",
        "paired_with", "met_on", "not_met_on", "campus_type", "county", "y_rating_final"]


# ------------------------------------------------------------------- campus summaries

def _num(s: str | None) -> float:
    if s is None:
        return np.nan
    s = s.replace(",", "").replace("%", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def _grade(g: str) -> float:
    g = g.strip().upper()
    return float(_GRADE[g]) if g in _GRADE else (float(g) if g.isdigit() else np.nan)


def _lists(lines: list[str], start: int) -> tuple[list[str], list[str]]:
    """The "Met Standards on" and "Did Not Meet Standards on" columns, which pdftotext
    lays out side by side starting on line ``start``."""
    col = lines[start].find("Did Not Meet")
    met, not_met = [], []
    for ln in lines[start + 1:start + 14]:
        if ln.strip().startswith("In 20") or "Performance Index Report" in ln:
            break
        for part, out in ((ln[:col], met), (ln[col:col + 45], not_met)):
            m = re.match(r"\s*-\s*(.+?)\s*$", part)
            if m and m.group(1).upper() != "NONE":
                out.append(m.group(1).strip())
    return met, not_met


def parse_summary(text: str) -> dict:
    """The fields of one campus Accountability Summary (pdftotext -layout output)."""
    lines = text.splitlines()
    r: dict = {}
    m = re.search(r"(\d{4}) Accountability Summary", text)
    r["year"] = int(m.group(1)) if m else np.nan
    m = re.search(r"^\s*(.+?) \((\d{9})\) - (.+?)\s*$", text, re.M)
    r["name"], r["campus"], r["district_name"] = (m.group(1), m.group(2), m.group(3)) if m \
        else ("", "", "")
    r["rating"] = ""
    for k, ln in enumerate(lines):
        if "Accountability Rating" in ln:
            for nxt in lines[k + 1:k + 4]:
                left = re.split(r"\s{3,}", nxt.strip())[0] if nxt.strip() else ""
                if left:
                    r["rating"] = left
                    break
            break
    m = re.search(r"Rating Change due to (.+?)\s{2,}", text)
    r["rating_change"] = m.group(1).strip() if m else ""
    m = re.search(r"paired with (.+?) \((\d{9})\)", text)
    r["paired_with"] = m.group(2) if m else ""
    start = next((k for k, ln in enumerate(lines) if "Did Not Meet Standards on" in ln), None)
    met, not_met = _lists(lines, start) if start is not None else ([], [])
    r["met_on"], r["not_met_on"] = ";".join(met), ";".join(not_met)
    # Targets: printed under each index's column. A column without a target (e.g. Index 2
    # for high schools in 2014) prints nothing, so each target is matched to the nearest
    # "Index k" column of the chart header rather than taken in order.
    for k in range(4):
        r[f"t{k + 1}"] = np.nan
    hdr = next((ln for ln in lines if re.search(r"Index 1\s+Index 2\s+Index 3\s+Index 4", ln)),
               None)
    if hdr is not None:
        centers = np.array([hdr.index(f"Index {k}") + 3.5 for k in (1, 2, 3, 4)])
        for ln in lines:
            for mt in re.finditer(r"\(Target Score\s*=\s*([0-9]+|N/A|n/a)\s*\)", ln):
                k = int(np.argmin(np.abs(centers - (mt.start() + mt.end()) / 2)))
                r[f"t{k + 1}"] = _num(mt.group(1))
    for k, label in ((1, "Student Achievement"), (2, "Student Progress"),
                     (3, "Closing Performance Gaps")):
        m = re.search(rf"^\s*{k} - {label}\s+([\d,]+|N/A)\s+([\d,]+|N/A)\s+(\d+|N/A)", text,
                      re.M)
        r[f"p{k}"], r[f"m{k}"], r[f"s{k}"] = (_num(m.group(1)), _num(m.group(2)),
                                              _num(m.group(3))) if m else (np.nan,) * 3
    # Index 4: 2013 has a points / maximum / score row; 2014 on list the components and put
    # the index score at the end of the last component line.
    m = re.search(r"^\s*4 - Postsecondary Readiness\s+([\d,.]+|N/A)\s+([\d,.]+|N/A)\s+(\d+|N/A)",
                  text, re.M)
    r["p4"], r["m4"] = (_num(m.group(1)), _num(m.group(2))) if m else (np.nan, np.nan)
    s4 = _num(m.group(3)) if m else np.nan
    for comp, key in (("STAAR", "c4_staar"), ("Graduation Rate", "c4_grad"),
                      ("Graduation Plan", "c4_plan")):
        m = re.search(rf"^\s*{comp} Score\s+([\d.]+|N/A)", text, re.M)
        r[key] = _num(m.group(1)) if m else np.nan
    m = re.search(r"^\s*Postsecondary (?:Component|Indicator) Score\s+([\d.]+|N/A)\s+(\d+|N/A)\b",
                  text, re.M)
    if m:
        r["c4_post"] = _num(m.group(1))
        s4 = _num(m.group(2))
    else:
        r["c4_post"] = np.nan
    r["s4"] = s4
    # The bar chart's four rounded index scores (a cross-check of the table).
    chart = [np.nan] * 4
    for k, ln in enumerate(lines):
        if re.search(r"Index 1\s+Index 2\s+Index 3\s+Index 4", ln):
            for prev in reversed(lines[max(0, k - 4):k]):
                mm = re.match(r"\s*(\d+|N/A)\s+(\d+|N/A)\s+(\d+|N/A)\s+(\d+|N/A)\b", prev)
                if mm:
                    chart = [_num(x) for x in mm.groups()]
                    break
            break
    for k in range(4):
        r[f"chart{k + 1}"] = chart[k]
    m = re.search(r"Campus Type\s+(.+?)\s*$", text, re.M)
    r["campus_type"] = m.group(1) if m else ""
    m = re.search(r"Campus Size\s+([\d,]+) Students?", text)
    r["n_students"] = _num(m.group(1)) if m else np.nan
    m = re.search(r"Grade Span\s+(\S+)\s*-\s*(\S+)", text)
    r["low_grade"], r["high_grade"] = (_grade(m.group(1)), _grade(m.group(2))) if m \
        else (np.nan, np.nan)
    for pat, key in ((r"Disadvantaged\s+([\d.]+%?|N/A)", "econ"),
                     (r"Learners\s+([\d.]+%?|N/A)", "ell"),
                     (r"Mobility Rate\s+([\d.]+%?|N/A)", "mobility"),
                     (r"Special\s*\n?.*?Education\s+([\d.]+%?|N/A)", "sped")):
        m = re.search(pat, text)
        r[key] = _num(m.group(1)) if m else np.nan
    return r


def _read_summaries(year: int) -> pd.DataFrame:
    d = SUMMARIES / str(year)
    if not d.exists():
        raise FileNotFoundError(
            f"{d} missing. Run python -m experiments.datasets.tx_index_2013.download")
    rows = []
    for f in sorted(d.glob("c*.txt")):
        r = parse_summary(f.read_text("utf-8"))
        r["file_campus"] = f.stem[1:]
        rows.append(r)
    df = pd.DataFrame(rows)
    bad = df.campus != df.file_campus
    if bad.any():
        raise ValueError(f"{year}: {int(bad.sum())} summaries name another campus")
    return df.drop(columns="file_campus")


# ------------------------------------------------------------- lists and directory

def read_campus_list(year: int) -> pd.DataFrame:
    """TEA's final campus ratings list for the year (after appeals): the rating section
    each campus is listed under and the note ("Appeal": changed on appeal; "Data": rating
    changed for data integrity)."""
    path = RAW / f"lists/campuslist_{year}.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run the download script.")
    rating, rows = "", []
    for ln in path.read_text("utf-8").splitlines():
        m = re.search(r"CAMPUS RATING:\s*(.+?)\s*$", ln)
        if m:
            rating = _RATINGS.get(m.group(1).upper(), m.group(1))
            continue
        m = re.search(r"\b(\d{9})\b(.*)$", ln)
        if m and rating:
            note = ("Appeal" if re.search(r"\bAppeal\b", m.group(2)) else
                    "Data Integrity" if re.search(r"\bData\b", m.group(2)) else "")
            rows.append({"campus": m.group(1), "rating_final": rating, "note": note})
    df = pd.DataFrame(rows).drop_duplicates("campus")
    df["year"] = year
    return df


def read_directory(year: int) -> pd.DataFrame:
    """Campus directory behind TEA's search page: campus, district, ESC region, county."""
    js = (RAW / f"lists/campdist_{year}.js").read_text("latin-1")
    body = js.split("var camp", 1)[1].split("var ", 1)[0]
    rows = re.findall(r'\["(\d{9})", "(\d{6})", "[^"]*", "[^"]*", "(\d+)", "([^"]*)"\]', body)
    df = pd.DataFrame(rows, columns=["campus", "district", "region", "county"])
    df["region"] = df.region.astype(int)
    df["year"] = year
    return df.drop_duplicates("campus")


# ---------------------------------------------------------------------------- rule

def _margin(points: pd.Series, maximum: pd.Series, target: pd.Series) -> pd.Series:
    """100 * points / maximum - (target - 0.5): >= 0 exactly when the rounded index score
    meets the target (TEA rounds half up)."""
    return 100 * points / maximum.where(maximum > 0) - (target - 0.5)


def _meets(points: pd.Series, maximum: pd.Series, target: pd.Series) -> pd.Series:
    """Exact integer test of round-half-up(100 * points / maximum) >= target."""
    ok = points.notna() & maximum.notna() & target.notna() & (maximum > 0)
    out = pd.Series(np.nan, index=points.index)
    out[ok] = (200 * points[ok] >= (2 * target[ok] - 1) * maximum[ok]).astype(float)
    return out


def state_rule(df: pd.DataFrame) -> pd.DataFrame:
    """The rating computed from the published points and targets, with each step kept."""
    out = pd.DataFrame(index=df.index)
    for k in (1, 2, 3):
        out[f"u{k}"] = 100 * df[f"p{k}"] / df[f"m{k}"].where(df[f"m{k}"] > 0)
        out[f"q{k}"] = _margin(df[f"p{k}"], df[f"m{k}"], df[f"t{k}"])
        out[f"met{k}"] = _meets(df[f"p{k}"], df[f"m{k}"], df[f"t{k}"])
        # Is the published score round-half-up of the ratio?
        out[f"round_ok{k}"] = (np.floor(out[f"u{k}"] + 0.5 + EPS) == df[f"s{k}"]).astype(float) \
            .where(out[f"u{k}"].notna())
    # Index 4. Elementary and middle schools (and any campus without all four components):
    # the STAAR component (one decimal), rounded half up. Otherwise the published score.
    staar_only = df.c4_grad.isna() & df.c4_plan.isna() & df.c4_post.isna() & df.p4.isna()
    u4 = df.c4_staar.where(staar_only)
    u4 = u4.fillna(100 * df.p4 / df.m4.where(df.m4 > 0))
    u4 = u4.fillna(df.s4)                       # four components: whole-number score only
    # Index 4 counts only when the summary prints an Index 4 score: some small high schools
    # show a STAAR component but no Index 4 score, and are not evaluated on Index 4.
    u4 = u4.where(df.s4.notna())
    out["u4"] = u4
    out["q4"] = u4 - (df.t4 - 0.5)
    met4 = (np.round(10 * u4) >= 10 * df.t4 - 5).astype(float)
    out["met4"] = met4.where(u4.notna() & df.t4.notna())
    out["i4_staar_only"] = staar_only.astype(float)
    or_year = df.year.isin(OR_YEARS)
    # Index 1 / Index 2 part: AND (2013-2014) or OR (2015 on); either one alone if only one.
    q12_or = np.fmax(out.q1, out.q2)
    q12_and = np.fmin(out.q1, out.q2)
    out["q12"] = np.where(or_year, q12_or, q12_and)
    m12_or = np.fmax(out.met1, out.met2)
    m12_and = np.fmin(out.met1, out.met2)
    out["met12"] = np.where(or_year, m12_or, m12_and)
    met = pd.Series(out.met12, index=df.index)
    for k in (3, 4):
        met = np.fmin(met, out[f"met{k}"])            # NaN (no data) is ignored by fmin
    out["met_computed"] = met
    q = pd.Series(out.q12, index=df.index)
    for k in (3, 4):
        q = np.fmin(q, out[f"q{k}"])
    out["q_min"] = q
    # Which required margin binds (is smallest) for each campus.
    parts = pd.DataFrame({"12": out.q12, "3": out.q3, "4": out.q4})
    out["binding"] = parts.idxmin(axis=1, skipna=True).where(parts.notna().any(axis=1), "")
    return out


# --------------------------------------------------------------------------- panel

def build() -> pd.DataFrame:
    """Every campus with a static summary in 2013, 2014, 2015 or 2017, with the computed
    rule, TEA's final rating, directory fields, district aggregates, the next summary
    year's Index 1 and the IR history from the campus lists (2013-2017)."""
    if not CACHE.exists():
        df = _build()
        PROCESSED.mkdir(parents=True, exist_ok=True)
        tmp = CACHE.with_name(CACHE.name + ".part")
        df.to_csv(tmp, index=False, compression="gzip")
        tmp.replace(CACHE)
    p = pd.read_csv(CACHE, keep_default_na=False, na_values=[""], low_memory=False,
                    dtype={c: str for c in _TXT})
    for c in _TXT:
        if c in p:
            p[c] = p[c].fillna("").astype(str)
    return p


def _build() -> pd.DataFrame:
    p = pd.concat([_read_summaries(y) for y in SUMMARY_YEARS], ignore_index=True)
    p["district"] = p.campus.str[:6]
    lists = pd.concat([read_campus_list(y) for y in YEARS], ignore_index=True)
    dirs = pd.concat([read_directory(y) for y in YEARS], ignore_index=True)
    p = p.merge(lists, on=["year", "campus"], how="left")
    p = p.merge(dirs[["year", "campus", "region", "county"]], on=["year", "campus"], how="left")
    p = pd.concat([p, state_rule(p)], axis=1)
    p["charter"] = (p.district.str[3] == "8").astype(float)
    p["is_paired"] = (p.paired_with != "").astype(float)
    # AEA campuses: Met Alternative Standard, or an Index 1 target other than the non-AEA
    # one (AEA targets are 25, 30, 35, 35, 35).
    p["is_aea"] = ((p.rating_final == "Met Alternative Standard")
                   | (p.t1.notna() & (p.t1 != p.year.map(INDEX1_TARGET)))).astype(float)
    # District aggregates over every campus summary of the district-year.
    w = p.n_students.fillna(0)
    g = pd.DataFrame({"year": p.year, "district": p.district, "n": w,
                      "econ_n": w * p.econ.fillna(0), "has": w * p.econ.notna()})
    agg = g.groupby(["year", "district"], as_index=False)[["n", "econ_n", "has"]].sum()
    agg["district_n_students"] = agg.n
    agg["district_econ"] = agg.econ_n / agg.has.where(agg.has > 0)
    p = p.merge(agg[["year", "district", "district_n_students", "district_econ"]],
                on=["year", "district"], how="left")
    # Next summary year's Index 1 (unrounded and published), same campus number.
    p["y_year"] = p.year.map(NEXT)
    nxt = p[["campus", "year", "u1", "s1", "q1", "rating_final", "is_paired", "is_aea",
             "campus_type"]].rename(columns={
        "year": "y_year", "u1": "y_index1", "s1": "y_index1_published", "q1": "y_q1",
        "rating_final": "y_rating_final", "is_paired": "y_paired", "is_aea": "y_aea",
        "campus_type": "y_campus_type"})
    p = p.merge(nxt, on=["campus", "y_year"], how="left")
    p.loc[p.y_paired == 1, ["y_index1", "y_index1_published"]] = np.nan
    p["y_gain"] = p.y_index1 - p.u1           # change in unrounded Index 1
    # 2017 ratings: the next year is the first A-F year. Y = the 2018 Student Achievement
    # domain scaled score from the tx_school_ratings dataset (its reader, not a copy).
    p["y_sa2018"] = np.nan
    try:
        from experiments.datasets.tx_school_ratings.adapter import build as build_af
        af = build_af()
        af = af[(af.year == 2018) & af.sa.notna()][["campus", "sa"]].drop_duplicates("campus")
        m17 = p.year == 2017
        p.loc[m17, "y_sa2018"] = p.loc[m17, "campus"].map(af.set_index("campus").sa)
    except FileNotFoundError:
        pass
    # IR history from the final campus lists (2013-2017, including 2016).
    ir = lists.assign(ir=(lists.rating_final == "Improvement Required").astype(float))
    wide = ir.pivot(index="campus", columns="year", values="ir")
    for y in YEARS:
        p[f"ir_{y}"] = p.campus.map(wide[y]) if y in wide else np.nan
    streak = []
    for camp, yr in zip(p.campus, p.year):
        n = 0
        for y in range(int(yr), 2012, -1):
            v = wide.at[camp, y] if (camp in wide.index and y in wide.columns) else np.nan
            if v == 1:
                n += 1
            else:
                break
        streak.append(n)
    p["ir_streak"] = streak            # consecutive IR years up to and including this year
    p["ir_next_list"] = [wide.at[c, y + 1] if (c in wide.index and y + 1 in wide.columns)
                         else np.nan for c, y in zip(p.campus, p.year)]
    return p.sort_values(["year", "campus"]).reset_index(drop=True)


# -------------------------------------------------------------------------- checks

def rule_check() -> pd.DataFrame:
    """Per year, for non-AEA campuses with their own index data and a Met Standard / IR
    rating: rounding of each published index score; the computed rating against the
    summary's rating and TEA's final list; and the computed index results against the
    summary's "Did Not Meet Standards on" list (which lists each missed index, also a
    missed Index 1 or 2 under the OR rule)."""
    p = build()
    rows = []
    for y, g in p.groupby("year"):
        rated = g.rating_final.isin(["Met Standard", "Improvement Required"])
        own = (g.is_aea == 0) & (g.is_paired == 0) & rated & g.met_computed.notna()
        h = g[own]
        ir_comp = h.met_computed == 0
        ir_summ = h.rating == "Improvement Required"
        ir_fin = h.rating_final == "Improvement Required"
        idx_agree = []
        for k, label in ((1, "Student Achievement"), (2, "Student Progress"),
                         (3, "Closing Performance Gaps"), (4, "Postsecondary Readiness")):
            have = h[f"met{k}"].notna()
            listed = h.not_met_on.str.contains(label, regex=False)
            miss = h[f"met{k}"] == 0          # listed per index, also under the OR rule
            idx_agree.append(round(float((miss == listed)[have].mean()), 5))
        rows.append({
            "year": y, "summaries": len(g), "aea": int((g.is_aea == 1).sum()),
            "paired": int((g.is_paired == 1).sum()),
            "own_rated_nonaea": int(own.sum()),
            "round_ok_1": float(h.round_ok1.mean()), "round_ok_2": float(h.round_ok2.mean()),
            "round_ok_3": float(h.round_ok3.mean()),
            "ir_computed": int(ir_comp.sum()), "ir_summary": int(ir_summ.sum()),
            "ir_final": int(ir_fin.sum()),
            "agree_summary": float((ir_comp == ir_summ).mean()),
            "agree_final": float((ir_comp == ir_fin).mean()),
            "appeals": int((h.note == "Appeal").sum()),
            "index_list_agree": idx_agree,
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------- samples

def _design(df: pd.DataFrame) -> tuple[np.ndarray, list]:
    X = pd.DataFrame(index=df.index)

    def fill(name, v):
        X[name] = v.fillna(v.median())
        if v.isna().any():
            X[f"{name}_missing"] = v.isna().astype(float)

    fill("pct_econ_dis", df.econ)
    fill("pct_ell", df.ell)
    fill("mobility", df.mobility)
    fill("log_students", np.log(df.n_students.where(df.n_students > 0)))
    fill("district_log_students",
         np.log(df.district_n_students.where(df.district_n_students > 0)))
    fill("district_pct_econ_dis", df.district_econ)
    fill("low_grade", df.low_grade)
    fill("high_grade", df.high_grade)
    X["middle"] = (df.campus_type == "Middle School").astype(float)
    X["charter"] = df.charter.astype(float)
    for y in sorted(df.year.unique())[1:]:
        X[f"year_{y}"] = (df.year == y).astype(float)
    for r in sorted(df.region.dropna().unique()):
        if r != 4:                                   # Region 4 (Houston) is the base
            X[f"region_{int(r):02d}"] = (df.region == r).astype(float)
    X = X.loc[:, X.std() > 0]
    return X.to_numpy(float), list(X.columns)


def _frame(years=MAIN_YEARS, y_col: str = "y_index1") -> pd.DataFrame:
    """Non-AEA, unpaired elementary and middle schools rated Met Standard or IR, with
    Index 1 and 3 (and Index 4 from 2014). Index 2 may be missing (no prior-year scores,
    e.g. a grade-3-only campus); then M12 = m1, as the rule counts only indexes with data."""
    p = build()
    need_i4 = p.year >= 2014
    m = (p.year.isin(years) & p.campus_type.isin(MAIN_TYPES) & (p.is_aea == 0)
         & (p.is_paired == 0) & p.rating_final.isin(["Met Standard", "Improvement Required"])
         & p.q1.notna() & p.q3.notna() & (p.q4.notna() | ~need_i4)
         & p[y_col].notna())
    return p[m].reset_index(drop=True)


def _extras(df: pd.DataFrame) -> dict:
    return {
        "year": df.year.to_numpy(int), "campus": df.campus.to_numpy(str),
        "district": df.district.to_numpy(str), "region": df.region.to_numpy(float),
        "campus_type": df.campus_type.to_numpy(str),
        "margins": df[["q1", "q2", "q3", "q4"]].to_numpy(float),
        "m12": df.q12.to_numpy(float),
        "index_unrounded": df[["u1", "u2", "u3", "u4"]].to_numpy(float),
        "index_published": df[["s1", "s2", "s3", "s4"]].to_numpy(float),
        "targets": df[["t1", "t2", "t3", "t4"]].to_numpy(float),
        "binding": df.binding.to_numpy(str),
        "d_full": (df.met_computed == 0).to_numpy(float),
        "rating_published": df.rating_final.to_numpy(str),
        "eligible_published": (df.rating_final == "Improvement Required").to_numpy(float),
        "appeal": (df.note == "Appeal").to_numpy(float),
        "rating_note": df.note.to_numpy(str),          # "Appeal", "Data Integrity" or ""
        "or_rule_year": df.year.isin(OR_YEARS).to_numpy(float),
        "ir_streak": df.ir_streak.to_numpy(float),
        "ir_next": df.ir_next_list.to_numpy(float),
        "y_year": df.y_year.to_numpy(float),
        "index1_now": df.u1.to_numpy(float),
        "n_tests": df.m1.to_numpy(float),       # Index 1 maximum points = tests
    }


def _rule_text(years) -> str:
    and_years = [y for y in years if y not in OR_YEARS]
    or_years = [y for y in years if y in OR_YEARS]
    parts = []
    if and_years:
        parts.append(f"{and_years}: Met Standard iff m1, m2, m3 (and m4 from 2014) >= 0")
    if or_years:
        parts.append(f"{or_years}: Met Standard iff max(m1, m2), m3, m4 >= 0")
    return "; ".join(parts) + ". D = 1 - Met Standard (Improvement Required)"


_Y = {"y_index1": "next rated year's Index 1 (unrounded)",
      "y_gain": "next rated year's Index 1 minus this year's (unrounded)",
      "y_sa2018": "2018 Student Achievement domain scaled score"}
_CITE = ("Texas Education Agency, Accountability Summaries 2013-2017 (static campus "
         "reports on rptsvr1.tea.texas.gov) and Accountability Manuals 2013-2017")
# Two-score views: column names in the panel, and what they mean.
PAIRS = {"12_34": (("q12", "q34"), "(Index 1/2 margin, min(Index 3, Index 4) margin)"),
         "12_3": (("q12", "q3"), "(Index 1/2 margin, Index 3 margin)"),
         "1_3": (("q1", "q3"), "(Index 1 margin, Index 3 margin)")}
DEFAULT_PAIR = "12_34"


def _either_below(Q: np.ndarray) -> np.ndarray:
    return (Q.min(axis=1) < -EPS).astype(int)


def _sample(view: str, years, y_col: str, pair: str = DEFAULT_PAIR) -> RDDSample:
    df = _frame(years, y_col)
    df["q34"] = np.fmin(df.q3, df.q4)              # q4 is NaN (not required) in 2013
    X, names = _design(df)
    ex = _extras(df)
    ex.update({"rule_text": _rule_text(years), "side": "below", "y_definition": _Y[y_col]})
    yrs = ", ".join(map(str, years))
    if view == "min":
        Q, thr, rule = df.q_min.to_numpy(float), 0.0, (lambda Q: (Q < -EPS).astype(int))
        name, qdesc = f"tx_index_2013_{y_col}", "Q = smallest required margin"
        ex.update({"cutoff": 0.0})
    else:
        cols, qdesc = PAIRS[pair] if view == "two" else (
            ("q12", "q3", "q4"), "(Index 1/2 margin, Index 3 margin, Index 4 margin)")
        Q = df[list(cols)].to_numpy(float)
        thr, rule = tuple(0.0 for _ in cols), _either_below
        name = f"tx_index_2013_{view}_score_{pair if view == 'two' else '12_3_4'}_{y_col}"
        ex.update({"cutoffs": thr, "rule": "either", "q_columns": cols,
                   "d_pair_vs_full": float((rule(Q) == ex["d_full"]).mean())})
    return RDDSample(
        Q=Q, X=X, Y=df[y_col].to_numpy(float), threshold=thr, name=name, feature_names=names,
        description=(f"Texas campus accountability (four-index system), elementary and middle "
                     f"schools, rating years {yrs}. {qdesc}, each margin = unrounded index - "
                     "(target - 0.5); treatment = 1{a margin < 0} (Improvement Required); "
                     f"Y = {_Y[y_col]}."),
        citation=_CITE, treatment_rule=rule, extras=ex)


def load(years=MAIN_YEARS) -> RDDSample:
    """Single-score view: Q = min(M12, m3, m4), the smallest required margin in index
    points (M12 = min(m1, m2) in 2013-2014, max(m1, m2) from 2015; m4 not required in
    2013). D = 1{Q < 0} = Improvement Required; the treated side is below the cutoff.
    Y = next rated year's Index 1, unrounded. ``years=(2013, 2014, 2015)`` adds 2015 with
    Y = 2017 Index 1 (``extras["y_year"]``)."""
    return _sample("min", years, "y_index1")


def load_gain(years=MAIN_YEARS) -> RDDSample:
    """As ``load()``, with Y = next rated year's Index 1 minus this year's (unrounded): the
    same jump at the cutoff, less noise from campus level."""
    return _sample("min", years, "y_gain")


def load_two_score(years=MAIN_YEARS, pair: str = DEFAULT_PAIR) -> RDDSample:
    """Two-score view, D = 1{Q1 < 0 or Q2 < 0} = Improvement Required (cutoffs 0, rule
    "either": IR when either margin is below its cutoff; Met Standard needs both).
    ``pair="12_34"`` (default): Q = (M12, min(m3, m4)), which reproduces the full rule.
    ``"12_3"``: (M12, m3); ``"1_3"``: (m1, m3). ``extras["d_pair_vs_full"]`` is the share
    of campuses whose D equals the full rule's."""
    return _sample("two", years, "y_index1", pair)


def load_three_score(years=MAIN_YEARS) -> RDDSample:
    """Q = (M12, m3, m4), D = 1{any margin < 0}. Exact rule from 2014 on; in 2013 m4 is
    NaN for every campus (Index 4 not evaluated for elementary and middle schools), so the
    default drops 2013."""
    return _sample("three", tuple(y for y in years if y >= 2014), "y_index1")


def load_2017(view: str = "min", pair: str = DEFAULT_PAIR) -> RDDSample:
    """2017 ratings (the "Index 1 OR Index 2" rule). The next year is the first A-F year,
    so Y = the campus's 2018 Student Achievement domain scaled score (tx_school_ratings).
    ``view``: "min", "two" or "three"."""
    return _sample(view, (2017,), "y_sa2018", pair)
