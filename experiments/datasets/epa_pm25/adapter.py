"""PM2.5 nonattainment designations under the Clean Air Act (EPA): a county with a monitor
whose design value exceeds the NAAQS is designated nonattainment and must adopt controls.

Under CAA section 107(d), EPA designates every area after it sets or revises a NAAQS. A
monitor site's design value (DV) is a 3-year statistic (40 CFR Part 50, Appendix N):

* annual DV: the 3-year mean of annual means (each the mean of four quarterly means),
  rounded to 0.1 ug/m3;
* 24-hour DV: the 3-year mean of annual 98th percentiles, rounded to 1 ug/m3.

A site violates a standard when its DV is above the level, and "if a PM2.5 monitor was
violating the standard ..., at a minimum we designated the entire county where that
monitor is located as nonattainment" (EPA, 70 FR 944, preamble section VII). EPA then
adds nearby counties that contribute to the violation (metro-area presumption, nine-factor
analysis), so the designation is fuzzy below the cutoff. Three rounds:

    round  standard(s)                   DV years   effective         Green Book NAAQS
    1997   annual > 15.0 or 24-h > 65    2001-2003  2005-04-05        PM-2.5 (1997)
    2006   24-h > 35                     2006-2008  2009-12-14        PM-2.5 (2006)
    2012   annual > 12.0                 2011-2013  2015-04-15        PM-2.5 (2012)

(70 FR 944 and 70 FR 19844; 74 FR 58688; 80 FR 2206 and 80 FR 18535.) In the 1997 and 2012
rounds EPA withdrew some nonattainment designations before the effective date because the
next 3-year period (2002-2004, 2012-2014) met the standard: eight areas in April 2005 and
five in April 2015 (extras["pre_effective_reversal"]).

Q = county maximum of the complete site DVs of the round's DV period, from EPA's published
    design-value reports, minus the cutoff. Because the DV is rounded (half up), "DV above
    the level" is exactly "DV >= level + half a rounding unit": the cutoffs are 15.05
    (1997 annual), 65.5 (1997 24-hour), 35.5 (2006) and 12.05 (2012). D = 1{Q >= 0}, the
    nonattainment side above. Sites EPA marks incomplete or not comparable to the NAAQS
    are dropped (EPA designated on complete DVs; see site_dvs()); in the 2012 round the
    DV is EPA's county summary, which differs from the site maximum only for Imperial, CA.
load():  the 1997 round, which has the most counties near its cutoff: Q = max annual DV
    2001-2003 - 15.05.
load_two_score(): the 1997 round, Q = (max annual DV, max 24-hour DV) in ug/m3, threshold
    (15.05, 65.5), D = 1 if either is at or above its cutoff ("either" rule).
load_2006(), load_2012(): the other rounds. load_pooled(): all three, Q as a share of the
    level, with round indicators. load_1997_dvsite(): Y at the county's DV site.
X = county characteristics that are not the period's pollution: log population and log
    density (the year before the DV period), manufacturing share of employment (County
    Business Patterns, same year), % urban (2010 Census), centroid latitude and longitude,
    EPA region, the earliest PM levels (county mean of 1999-2000 PM2.5 annual means and of
    1996-1998 PM10 annual means, with missing indicators), and for the later rounds the
    county's earlier PM2.5 nonattainment status.
Y = change in the annual mean PM2.5 at the county's monitors from the DV period to 3-5
    years after the designation (2008-2010, 2012-2014, 2018-2020): for each site with at
    least two annual means in each window, mean(post) - mean(DV period); Y is the mean over
    the county's sites (AQS parameter 88101, all data including event-flagged days).
    `y_change_early` (1-3 years after) and `y_change_dvsite` are alternatives.

Sample: counties with a complete DV for the round's standard and Y observed. Counties
designated without a monitor (contributing counties) are not in the sample.
"""
from __future__ import annotations

import os
import struct
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample
# Standard-library .xls/.xlsx reader (the environment has no openpyxl or xlrd), shared
# with the Florida dataset rather than copied.
from experiments.datasets.fl_school_grades._spreadsheet import read as read_sheet
from experiments.datasets.fl_school_grades._spreadsheet import sheet_names
from experiments.datasets.epa_pm25.download import (AQS_YEARS, CBP, DV_0608, DV_1113,
                                                    GAZETTEER, PHISTORY, POP_2000S,
                                                    POP_2010S, URBAN, aqs_annual)

HERE = Path(__file__).parent
PROCESSED = HERE / "data" / "processed"
MONITOR_CACHE = PROCESSED / "aqs_pm_site_years.csv.gz"
CACHE = PROCESSED / "epa_pm25_county_rounds.csv.gz"

# round -> rule and timing (Federal Register notices cited in the docstring)
ROUNDS = {
    1997: dict(naaqs="PM-2.5 (1997)", dv_years=(2001, 2002, 2003), designated=2005,
               post_years=(2008, 2009, 2010), cov_year=2000,
               levels={"annual": 15.0, "daily": 65.0}, main="annual"),
    2006: dict(naaqs="PM-2.5 (2006)", dv_years=(2006, 2007, 2008), designated=2009,
               post_years=(2012, 2013, 2014), cov_year=2005,
               levels={"daily": 35.0}, main="daily"),
    2012: dict(naaqs="PM-2.5 (2012)", dv_years=(2011, 2012, 2013), designated=2015,
               post_years=(2018, 2019, 2020), cov_year=2010,
               levels={"annual": 12.0}, main="annual"),
}
HALF_UNIT = {"annual": 0.05, "daily": 0.5}      # DVs are rounded to 0.1 and to 1 ug/m3
MAIN_ROUND = 1997
EPS = 1e-9
# The eight areas EPA moved from nonattainment to attainment before the April 5, 2005
# effective date (70 FR 19844, section II): seven on 2002-2004 data, and Columbus, GA-AL
# on 2002-2004 data with spatial averaging. County FIPS codes.
APRIL_2005_REVERSALS = {
    "13059": "Athens, GA", "18039": "Elkhart, IN", "18141": "Elkhart, IN",
    "21067": "Lexington, KY", "21167": "Lexington, KY",
    "54049": "Marion County, WV", "54061": "Marion County, WV", "54033": "Marion County, WV",
    "06073": "San Diego, CA", "39095": "Toledo, OH", "39173": "Toledo, OH",
    "39029": "Youngstown, OH-PA", "39099": "Youngstown, OH-PA", "39155": "Youngstown, OH-PA",
    "42085": "Youngstown, OH-PA",
    # Columbus, GA-AL: spatial averaging approved and 2002-2004 data attaining (15.04).
    "01113": "Columbus, GA-AL", "13215": "Columbus, GA-AL",
}
# Counties with a violating 2011-2013 DV whose January 2015 nonattainment designation EPA
# changed to unclassifiable/attainment (Louisville: unclassifiable) on 2012-2014 data
# before the April 15, 2015 effective date (80 FR 18535, section II.B).
APRIL_2015_REVERSALS = {
    "18019": "Louisville, KY-IN", "39017": "Cincinnati-Hamilton, OH-KY",
    "39061": "Cincinnati-Hamilton, OH-KY", "39151": "Canton-Massillon, OH",
    "42021": "Johnstown, PA", "42095": "Allentown, PA",
}
# Midpoints of the County Business Patterns employment-size flags (suppressed cells).
CBP_FLAG_MID = {"A": 10, "B": 60, "C": 175, "E": 375, "F": 750, "G": 1750, "H": 3750,
                "I": 7500, "J": 17500, "K": 37500, "L": 75000, "M": 150000}
# EPA regions by state FIPS.
EPA_REGION = {}
for _r, _states in {
        1: "09 23 25 33 44 50", 2: "34 36 72 78", 3: "10 11 24 42 51 54",
        4: "01 12 13 21 28 37 45 47", 5: "17 18 26 27 39 55", 6: "05 22 35 40 48",
        7: "19 20 29 31", 8: "08 30 38 46 49 56", 9: "04 06 15 32 60 66 69",
        10: "02 16 41 53"}.items():
    for _s in _states.split():
        EPA_REGION[_s] = _r


# ------------------------------------------------------------------------ helpers

def _need(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run python -m experiments.datasets.epa_pm25.download")
    return path


def _write_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp.csv.gz")
    os.close(fd)
    try:
        df.to_csv(tmp, index=False, compression="gzip")
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _num(s) -> pd.Series:
    return pd.to_numeric(pd.Series(s).astype(str).str.strip().replace({"": None}),
                         errors="coerce")


def _fips(st, co) -> pd.Series:
    st = pd.Series(st).astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    co = pd.Series(co).astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    return st.str.zfill(2) + co.str.zfill(3)


def read_dbf(path: Path) -> pd.DataFrame:
    """dBASE III table (the Green Book export) as strings, with the standard library."""
    b = _need(path).read_bytes()
    n, header_len, rec_len = struct.unpack_from("<IHH", b, 4)
    fields, o = [], 32
    while b[o] != 0x0D:
        fields.append((b[o:o + 11].split(b"\0")[0].decode("latin1"), b[o + 16]))
        o += 32
    rows = []
    for i in range(n):
        r = b[header_len + i * rec_len: header_len + (i + 1) * rec_len]
        if r[:1] == b"*":                       # deleted record
            continue
        p, row = 1, []
        for _, width in fields:
            row.append(r[p:p + width].decode("latin1").strip())
            p += width
        rows.append(row)
    return pd.DataFrame(rows, columns=[f for f, _ in fields])


def _sheet(path: Path, name_start: str, header_row: int) -> pd.DataFrame:
    names = sheet_names(_need(path))
    k = next(i for i, s in enumerate(names) if s.startswith(name_start))
    rows = read_sheet(path, k)
    hdr = [str(h).strip() if h is not None else f"col{j}" for j, h in enumerate(rows[header_row])]
    body = [r + [None] * (len(hdr) - len(r)) for r in rows[header_row + 1:]]
    return pd.DataFrame([r[:len(hdr)] for r in body], columns=hdr)


# ------------------------------------------------------------- published design values

def _site_listing_0608(period: str) -> pd.DataFrame:
    """A site listing from EPA's 2006-2008 DV report (AQS as of 2009-07-07): the
    '2001-2003' or '2006-2008' sheet. Status codes (sheet 'site listing definitions'):
    1 complete and violating, 2 complete and meeting, 3 incomplete but above the level,
    4 incomplete and not above, 5 not comparable to the NAAQS."""
    d = _sheet(DV_0608, f"{period} site listing", 0)
    y = period.replace("-", "")
    col = lambda stem: next(c for c in d.columns if c.startswith(f"{stem}_{y}"))
    out = pd.DataFrame({
        "site": d["site"].astype(str).str.strip(),
        "fips": _fips(d["st"], d["cou"]),
        "annual_dv": _num(d[col("dva")]), "annual_status": _num(d[col("dvstatusas")]),
        "daily_dv": _num(d[col("dvd")]), "daily_status": _num(d[col("dvstatusds")]),
    })
    return out[out.site.str.len() == 9].reset_index(drop=True)


def _site_listing_1113() -> pd.DataFrame:
    """Table 5 of EPA's 2011-2013 DV report (final 2014-08-28): valid and invalid site DVs
    in separate columns."""
    d = _sheet(DV_1113, "Table 5", 3)
    c = list(d.columns)
    out = pd.DataFrame({
        "site": d["Site ID"].astype(str).str.strip(),
        "fips": _fips(d["FIPS State"], d["FIPS County"]),
        "annual_dv": _num(d[c[17]]), "annual_dv_invalid": _num(d[c[18]]),
        "daily_dv": _num(d[c[24]]), "daily_dv_invalid": _num(d[c[25]]),
        "exclusion_comment": d[c[29]].fillna("").astype(str).str.strip(),
    })
    return out[out.site.str.len() == 9].reset_index(drop=True)


def site_dvs() -> pd.DataFrame:
    """Site x round design values with a validity flag for each standard of the round."""
    out = []
    # 1997 and 2006 rounds: complete DVs only (status 1 or 2). Appendix N also accepts an
    # incomplete DV above the level (status 3), but EPA did not designate on those: in
    # the 2006 round none of the 11 counties that violate only through status-3 sites was
    # designated, and in the 1997 round 5 of 13 were, all inside metro areas designated
    # for other counties' violations (README). The 24-hour status in the 2009 report is
    # against 35, so a 1997-round incomplete 24-hour DV above 65 would also count; there
    # is none.
    s = _site_listing_0608("2001-2003")
    s["annual_valid"] = s.annual_status.isin([1, 2])
    s["daily_valid"] = s.daily_status.isin([1, 2]) | (
        s.daily_status.isin([3, 4]) & (s.daily_dv >= 65.5 - EPS))
    out.append(s.assign(round=1997))
    s = _site_listing_0608("2006-2008")
    s["annual_valid"] = s.annual_status.isin([1, 2])
    s["daily_valid"] = s.daily_status.isin([1, 2])
    out.append(s.assign(round=2006))
    s = _site_listing_1113()
    s["annual_valid"] = s.annual_dv.notna()
    s["daily_valid"] = s.daily_dv.notna()
    s["annual_dv"] = s.annual_dv.fillna(s.annual_dv_invalid)
    s["daily_dv"] = s.daily_dv.fillna(s.daily_dv_invalid)
    out.append(s.assign(round=2012))
    return pd.concat(out, ignore_index=True)


def _county_dvs() -> pd.DataFrame:
    s = site_dvs()
    rows = []
    for (r, f), g in s.groupby(["round", "fips"]):
        row = {"round": r, "fips": f, "n_sites_listed": len(g)}
        for k in ("annual", "daily"):
            v = g[g[f"{k}_valid"]]
            row[f"{k}_dv"] = v[f"{k}_dv"].max() if len(v) else np.nan
            row[f"{k}_site"] = (v.loc[v[f"{k}_dv"].idxmax(), "site"]
                                if len(v) and v[f"{k}_dv"].notna().any() else "")
            row[f"n_{k}_valid"] = int(v[f"{k}_dv"].notna().sum())
            inv = g[~g[f"{k}_valid"]]
            row[f"{k}_dv_invalid_max"] = inv[f"{k}_dv"].max() if len(inv) else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    # 2012 round: EPA's county summary (Table 4) equals the maximum valid site DV in every
    # county but Imperial, CA, whose 14.3 uses 2011 and 2013 data "submitted to, but not
    # currently in, AQS" (the report's footnote 4; no site row). Use Table 4.
    t4 = _sheet(DV_1113, "Table 4", 3)
    c = t4.columns
    t4 = pd.Series(_num(t4[c[4]]).values, index=_fips(t4[c[0]], t4[c[1]]).values).dropna()
    k = (out["round"] == 2012) & out.fips.isin(t4.index)
    new = out.loc[k, "fips"].map(t4)
    changed = k & (np.abs(out["annual_dv"].fillna(-1) - out.fips.map(t4).fillna(-1)) > EPS)
    out.loc[changed, "annual_site"] = "county_summary"
    out.loc[k, "annual_dv"] = new
    return out


# ------------------------------------------------------------------- designations

def designations() -> pd.DataFrame:
    """Green Book phistory: for each PM2.5 NAAQS and county, the first year designated
    nonattainment and whether the whole county ('W') or part ('P') was designated."""
    ph = read_dbf(PHISTORY)
    ph = ph[ph.POLLUTANT.str.startswith("PM-2.5")]
    years = [c for c in ph.columns if c.startswith("PW_")]
    rows = []
    for _, r in ph.iterrows():
        ys = [int(c[3:]) for c in years if r[c] in ("P", "W")]
        if ys:
            rows.append({"naaqs": r.POLLUTANT, "fips": r.FIPS_STATE + r.FIPS_CNTY,
                         "first_year": min(ys), "part": r[f"PW_{min(ys)}"] == "P",
                         "county_name_gb": r.COUNTYNAME, "state": r.ST_ABBR})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- AQS monitor data

_AQS_COLS = ["State Code", "County Code", "Site Num", "Parameter Code", "POC",
             "Sample Duration", "Pollutant Standard", "Event Type", "Year",
             "Valid Day Count", "Completeness Indicator", "Arithmetic Mean",
             "98th Percentile", "Latitude", "Longitude"]


def _aqs_year(year: int) -> pd.DataFrame:
    """PM2.5 (88101, 'PM25 Annual 2012' rows: mean of quarterly means) and PM10 (81102,
    'PM10 24-hour 2006' rows: mean of daily values) monitor-years, with the annual mean
    including all event-flagged data ('No Events' or 'Events Included')."""
    with zipfile.ZipFile(_need(aqs_annual(year))) as z:
        with z.open(z.namelist()[0]) as fh:
            d = pd.read_csv(fh, dtype=str, usecols=_AQS_COLS)
    keep = (((d["Parameter Code"] == "88101") & (d["Pollutant Standard"] == "PM25 Annual 2012"))
            | ((d["Parameter Code"] == "81102")
               & (d["Pollutant Standard"] == "PM10 24-hour 2006")))
    d = d[keep & d["Event Type"].isin(["No Events", "Events Included"])]
    return pd.DataFrame({
        "year": year, "param": d["Parameter Code"].astype(int),
        "site": d["State Code"].str.zfill(2) + d["County Code"].str.zfill(3)
        + d["Site Num"].str.zfill(4),
        "poc": d["POC"].astype(int), "duration": d["Sample Duration"],
        "valid_days": pd.to_numeric(d["Valid Day Count"], errors="coerce"),
        "complete": d["Completeness Indicator"] == "Y",
        "mean": pd.to_numeric(d["Arithmetic Mean"], errors="coerce"),
        "lat": pd.to_numeric(d["Latitude"], errors="coerce"),
        "lon": pd.to_numeric(d["Longitude"], errors="coerce"),
    })


def site_years() -> pd.DataFrame:
    """One annual mean per site, parameter and year, 1996-2020: the POC with the most
    valid days, preferring filter-based 24-hour samples ('24 HOUR') over continuous
    monitors. Cached."""
    if not MONITOR_CACHE.exists():
        m = pd.concat([_aqs_year(y) for y in AQS_YEARS], ignore_index=True)
        m = m[m["mean"].notna()]
        m["filter"] = (m.duration == "24 HOUR").astype(int)
        m = m.sort_values(["site", "param", "year", "filter", "valid_days", "poc"],
                          ascending=[True, True, True, False, False, True])
        m = m.drop_duplicates(["site", "param", "year"])
        _write_atomic(m.drop(columns=["filter"]), MONITOR_CACHE)
    return pd.read_csv(MONITOR_CACHE, dtype={"site": str, "duration": str})


def _site_change(sy: pd.DataFrame, pre, post, min_years: int = 2) -> pd.DataFrame:
    """Per PM2.5 site: mean annual mean in each window, and the change, for sites with at
    least `min_years` annual means in each window."""
    p = sy[sy.param == 88101]
    a = p[p.year.isin(pre)].groupby("site")["mean"].agg(["mean", "size"])
    b = p[p.year.isin(post)].groupby("site")["mean"].agg(["mean", "size"])
    j = a.join(b, lsuffix="_pre", rsuffix="_post", how="inner")
    j = j[(j.size_pre >= min_years) & (j.size_post >= min_years)]
    j["change"] = j.mean_post - j.mean_pre
    j["fips"] = j.index.str[:5]
    return j.reset_index()


# ------------------------------------------------------------------- county covariates

def _population() -> pd.DataFrame:
    a = pd.read_csv(_need(POP_2000S), dtype={"STATE": str, "COUNTY": str}, encoding="latin1")
    a = a[a.SUMLEV == 50]
    b = pd.read_csv(_need(POP_2010S), dtype={"STATE": str, "COUNTY": str}, encoding="latin1")
    b = b[b.SUMLEV == 50]
    out = pd.DataFrame({"fips": a.STATE.str.zfill(2) + a.COUNTY.str.zfill(3),
                        "pop2000": a.POPESTIMATE2000, "pop2005": a.POPESTIMATE2005})
    out = out.merge(pd.DataFrame({"fips": b.STATE.str.zfill(2) + b.COUNTY.str.zfill(3),
                                  "pop2010": b.POPESTIMATE2010}), on="fips", how="outer")
    return out


def _gazetteer() -> pd.DataFrame:
    with zipfile.ZipFile(_need(GAZETTEER)) as z:
        g = pd.read_csv(z.open(z.namelist()[0]), sep="\t", dtype={"GEOID": str},
                        encoding="latin1")
    g.columns = [c.strip() for c in g.columns]
    return pd.DataFrame({"fips": g.GEOID.str.zfill(5), "land_sqmi": g.ALAND_SQMI,
                         "lat": g.INTPTLAT, "lon": g.INTPTLONG})


def _urban() -> pd.DataFrame:
    u = pd.read_csv(_need(URBAN), dtype={"STATE": str, "COUNTY": str}, encoding="latin1")
    return pd.DataFrame({"fips": u.STATE.str.zfill(2) + u.COUNTY.str.zfill(3),
                         "pct_urban": pd.to_numeric(u.POPPCT_URBAN, errors="coerce")})


def _cbp(year: int) -> pd.DataFrame:
    """Manufacturing (NAICS 31-33) share of mid-March employment, County Business
    Patterns. Suppressed cells use the midpoint of their size class."""
    with zipfile.ZipFile(_need(CBP[year])) as z:
        with z.open(z.namelist()[0]) as fh:
            d = pd.read_csv(fh, dtype=str, usecols=["fipstate", "fipscty", "naics", "empflag",
                                                    "emp"])
    d = d[d.naics.isin(["------", "31----"])]
    emp = pd.to_numeric(d.emp, errors="coerce")
    flag = d.empflag.fillna("").str.strip()
    emp = emp.where(flag == "", flag.map(CBP_FLAG_MID))
    d = d.assign(emp=emp, fips=d.fipstate + d.fipscty)
    w = d.pivot_table(index="fips", columns="naics", values="emp", aggfunc="first")
    tot = w["------"]
    man = w.get("31----", pd.Series(0.0, index=w.index)).fillna(0.0)
    share = (100 * man / tot).where(tot > 0).clip(0, 100)    # midpoints can exceed the total
    return pd.DataFrame({"fips": w.index, "manuf_share": share.values,
                         "log_cbp_emp": np.log(tot.where(tot > 0)).values})


def _pm_lags(sy: pd.DataFrame) -> pd.DataFrame:
    """Earliest PM levels: county mean over sites of the 1999-2000 PM2.5 annual means and
    of the 1996-1998 PM10 annual means."""
    out = []
    for param, years, name in ((88101, (1999, 2000), "pm25_1999_2000"),
                               (81102, (1996, 1997, 1998), "pm10_1996_1998")):
        p = sy[(sy.param == param) & sy.year.isin(years)]
        s = p.groupby("site")["mean"].mean()
        c = s.groupby(s.index.str[:5]).mean().rename(name)
        out.append(c)
    return pd.concat(out, axis=1).rename_axis("fips").reset_index()


# -------------------------------------------------------------------------- panel

def build() -> pd.DataFrame:
    """County x round panel: published county max DVs, the rule, Green Book designations,
    outcomes and covariates. Cached."""
    if not CACHE.exists():
        _write_atomic(_build(), CACHE)
    return pd.read_csv(CACHE, dtype={"fips": str, "annual_site": str, "daily_site": str,
                                     "dv_site_main": str})


def _build() -> pd.DataFrame:
    cdv = _county_dvs()
    gb = designations()
    sy = site_years()
    pop, gaz, urb, lags = _population(), _gazetteer(), _urban(), _pm_lags(sy)
    out = []
    for r, spec in ROUNDS.items():
        p = cdv[cdv["round"] == r].copy()
        main = spec["main"]
        p["dv_main"] = p[f"{main}_dv"]
        p["dv_site_main"] = p[f"{main}_site"]
        p["level_main"] = spec["levels"][main]
        p["cut_main"] = spec["levels"][main] + HALF_UNIT[main]
        p["rule"] = 0.0
        for k, lev in spec["levels"].items():
            p["rule"] = np.maximum(p["rule"], (p[f"{k}_dv"] >= lev + HALF_UNIT[k] - EPS)
                                   .astype(float))
        p.loc[p[[f"{k}_dv" for k in spec["levels"]]].isna().all(axis=1), "rule"] = np.nan
        g = gb[gb.naaqs == spec["naaqs"]]
        first = g.set_index("fips")
        p["na_first_year"] = p.fips.map(first.first_year)
        p["na_published"] = (p.na_first_year == spec["designated"]).astype(float)
        p["na_later"] = (p.na_first_year > spec["designated"]).astype(float)
        p["na_part"] = p.fips.map(first.part).fillna(False).astype(float)
        p["pre_effective_reversal"] = (
            (p.fips.isin(APRIL_2005_REVERSALS) & (r == 1997))
            | (p.fips.isin(APRIL_2015_REVERSALS) & (r == 2012))).astype(float)
        # prior PM2.5 nonattainment (an earlier round), predetermined for this round
        for pr, name in ((1997, "prior_na_1997"), (2006, "prior_na_2006")):
            gp = gb[(gb.naaqs == ROUNDS[pr]["naaqs"])
                    & (gb.first_year <= ROUNDS[pr]["designated"] + 2)]
            p[name] = (p.fips.isin(gp.fips) & (r > pr)).astype(float)
        # outcomes
        ch = _site_change(sy, spec["dv_years"], spec["post_years"])
        cy = ch.groupby("fips").agg(y_change=("change", "mean"),
                                    y_pre=("mean_pre", "mean"), y_post=("mean_post", "mean"),
                                    n_sites_y=("change", "size"))
        p = p.merge(cy, left_on="fips", right_index=True, how="left")
        p["y_change_dvsite"] = p.dv_site_main.map(ch.set_index("site").change)
        d0 = spec["designated"]
        early = _site_change(sy, spec["dv_years"], (d0 + 1, d0 + 2, d0 + 3))
        p["y_change_early"] = p.fips.map(early.groupby("fips").change.mean())
        # covariates in the year before the DV period
        cyear = spec["cov_year"]
        p = p.merge(pop[["fips", f"pop{cyear}"]].rename(columns={f"pop{cyear}": "pop"}),
                    on="fips", how="left")
        p = p.merge(_cbp(cyear), on="fips", how="left")
        p["round"] = r
        out.append(p)
    df = pd.concat(out, ignore_index=True)
    df = df.merge(gaz, on="fips", how="left").merge(urb, on="fips", how="left")
    df = df.merge(lags, on="fips", how="left")
    df["epa_region"] = df.fips.str[:2].map(EPA_REGION)
    df["state_fips"] = df.fips.str[:2]
    return df.sort_values(["round", "fips"]).reset_index(drop=True)


# --------------------------------------------------------------------------- checks

def rule_check() -> pd.DataFrame:
    """By round: counties with a valid DV, rule-violating counties, published
    nonattainment counties, and the 2x2 of the rule against the Green Book designation."""
    p = build()
    rows = []
    for r, g in p.groupby("round"):
        g = g[g.dv_main.notna()]
        rule, pub = g.rule == 1, g.na_published == 1
        rows.append({
            "round": r, "counties": len(g), "rule": int(rule.sum()), "published": int(pub.sum()),
            "agree": float((rule == pub).mean()),
            "rule_and_pub": int((rule & pub).sum()), "rule_not_pub": int((rule & ~pub).sum()),
            "pub_not_rule": int((~rule & pub).sum()),
            "P(pub|rule)": float(pub[rule].mean()) if rule.any() else np.nan,
            "P(pub|not rule)": float(pub[~rule].mean()),
            "pub_without_valid_dv": int(((p["round"] == r) & p.dv_main.isna()
                                         & (p.na_published == 1)).sum()),
        })
    return pd.DataFrame(rows)


def boundary_mass(rel_width: float = 0.10) -> dict:
    """Counties with a valid DV within rel_width x the level of each round's cutoff."""
    p = build()
    out = {}
    for r, g in p.groupby("round"):
        q = g.dv_main - g.cut_main
        out[r] = int((np.abs(q) <= rel_width * g.level_main).sum())
    return out


def local_jump(sample: RDDSample, h: float, d: np.ndarray | None = None,
               col: int = 0, cluster: np.ndarray | None = None,
               covariates: bool = False) -> dict:
    """Local-linear jump in Y at the cutoff (triangular kernel, bandwidth h, separate
    slopes), with an HC1 standard error, or a cluster-robust one with `cluster`. With
    `covariates`, sample.X enters linearly (centred in the window). With `d`, also the
    jump in d (first stage) and the Wald ratio (delta-method se, ignoring the covariance
    of the two jumps). For a two-score sample, `col` picks the score."""
    Q = np.asarray(sample.Q, float)
    q = Q - sample.threshold if Q.ndim == 1 else Q[:, col] - sample.threshold[col]

    def fit(y):
        w = np.clip(1 - np.abs(q) / h, 0, None)
        k = (w > 0) & np.isfinite(y)
        t = (q >= 0).astype(float)
        Z = np.column_stack([np.ones(k.sum()), t[k], q[k], t[k] * q[k]])
        if covariates:
            Xk = sample.X[k]
            Xk = Xk[:, Xk.std(axis=0) > 0]
            Z = np.column_stack([Z, Xk - Xk.mean(axis=0)])
        W = w[k]
        A = Z.T @ (Z * W[:, None])
        beta = np.linalg.solve(A, Z.T @ (W * y[k]))
        e = y[k] - Z @ beta
        n, m = Z.shape
        s = Z * (W * e)[:, None]
        if cluster is not None:
            s = pd.DataFrame(s).groupby(np.asarray(cluster)[k]).sum().to_numpy()
            g = len(s)
            adj = g / (g - 1) * (n - 1) / (n - m)
        else:
            adj = n / (n - m)
        V = np.linalg.solve(A, np.linalg.solve(A, s.T @ s).T) * adj
        return beta[1], np.sqrt(V[1, 1]), int(k.sum())

    b, se, n = fit(sample.Y)
    out = {"h": h, "n_in_window": n, "jump": float(b), "se": float(se),
           "covariates": covariates}
    if d is not None:
        fb, fse, _ = fit(np.asarray(d, float))
        out.update({"first_stage": float(fb), "first_stage_se": float(fse),
                    "wald": float(b / fb) if fb else np.nan,
                    "wald_se": float(np.sqrt(se ** 2 / fb ** 2 + b ** 2 * fse ** 2 / fb ** 4))
                    if fb else np.nan})
    return out


# -------------------------------------------------------------------------- samples

def _design(df: pd.DataFrame, rounds) -> tuple[np.ndarray, list]:
    X = pd.DataFrame(index=df.index)
    lp = np.log(df["pop"].where(df["pop"] > 0))
    X["log_pop"] = lp.fillna(lp.median())
    ld = lp - np.log(df.land_sqmi.where(df.land_sqmi > 0))
    X["log_density"] = ld.fillna(ld.median())
    X["covariates_missing"] = (lp.isna() | ld.isna()).astype(float)
    X["manuf_share"] = df.manuf_share.fillna(df.manuf_share.median())
    X["manuf_missing"] = df.manuf_share.isna().astype(float)
    X["pct_urban"] = df.pct_urban.fillna(df.pct_urban.median())
    X["lat"] = df.lat.fillna(df.lat.median())
    X["lon"] = df.lon.fillna(df.lon.median())
    for c in ("pm25_1999_2000", "pm10_1996_1998"):
        X[c] = df[c].fillna(df[c].median())
        X[f"{c}_missing"] = df[c].isna().astype(float)
    X["prior_na_1997"] = df.prior_na_1997
    X["prior_na_2006"] = df.prior_na_2006
    for reg in range(1, 11):
        if reg != 5:                                   # Region 5 (Great Lakes) is the base
            X[f"epa_region_{reg}"] = (df.epa_region == reg).astype(float)
    if len(rounds) > 1:
        for r in rounds[1:]:
            X[f"round_{r}"] = (df["round"] == r).astype(float)
    X = X.loc[:, X.std() > 0]
    return X.to_numpy(float), list(X.columns)


def _frame(rounds, y_col: str = "y_change", need: tuple = ()) -> pd.DataFrame:
    p = build()
    df = p[p["round"].isin(rounds) & p.dv_main.notna() & p[y_col].notna()]
    for c in need:
        df = df[df[c].notna()]
    return df.reset_index(drop=True)


def _extras(df: pd.DataFrame) -> dict:
    return {
        "fips": df.fips.to_numpy(str), "county_id": df.fips.to_numpy(str),
        "state_fips": df.state_fips.to_numpy(str), "round": df["round"].to_numpy(int),
        "eligible_published": df.na_published.to_numpy(float),
        "na_part": df.na_part.to_numpy(float), "na_later": df.na_later.to_numpy(float),
        "pre_effective_reversal": df.pre_effective_reversal.to_numpy(float),
        "dv_main": df.dv_main.to_numpy(float), "dv_site_main": df.dv_site_main.to_numpy(str),
        "annual_dv": df.annual_dv.to_numpy(float), "daily_dv": df.daily_dv.to_numpy(float),
        "n_sites_listed": df.n_sites_listed.to_numpy(int),
        "y_pre": df.y_pre.to_numpy(float), "y_post": df.y_post.to_numpy(float),
        "y_change_dvsite": df.y_change_dvsite.to_numpy(float),
        "n_sites_y": df.n_sites_y.to_numpy(float),
    }


def _round_sample(r: int, y_col: str = "y_change") -> RDDSample:
    spec = ROUNDS[r]
    main = spec["main"]
    cut = spec["levels"][main] + HALF_UNIT[main]
    df = _frame((r,), y_col)
    X, names = _design(df, (r,))
    lev = spec["levels"][main]
    return RDDSample(
        Q=(df.dv_main - cut).to_numpy(float), X=X, Y=df[y_col].to_numpy(float),
        threshold=0.0, name=f"epa_pm25_{r}_{y_col}", feature_names=names,
        description=(f"PM2.5 nonattainment designations, {r} NAAQS round. Q = county max "
                     f"valid {main} design value {spec['dv_years'][0]}-{spec['dv_years'][-1]}"
                     f" minus {cut} (DV above {lev}); treatment = 1{{Q >= 0}} (a violating "
                     f"monitor in the county; nonattainment side above); Y = {y_col}: change "
                     f"in annual mean PM2.5 at the county's monitors from "
                     f"{spec['dv_years'][0]}-{spec['dv_years'][-1]} to "
                     f"{spec['post_years'][0]}-{spec['post_years'][-1]}."),
        citation=("EPA PM2.5 design value reports (2006-2008, 2011-2013), EPA Green Book "
                  "(phistory), EPA AQS annual summaries; 70 FR 944, 74 FR 58688, 80 FR 2206"),
        treatment_rule=lambda Q: (Q >= -EPS).astype(int),
        extras={**_extras(df), "cutoff": cut, "level": lev, "round_spec": spec},
    )


def load() -> RDDSample:
    """1997 round, annual standard: Q = max annual DV 2001-2003 - 15.05, D = 1{Q >= 0}."""
    return _round_sample(MAIN_ROUND)


def load_1997() -> RDDSample:
    return _round_sample(1997)


def load_2006() -> RDDSample:
    """2006 24-hour round: Q = max 24-hour DV 2006-2008 - 35.5."""
    return _round_sample(2006)


def load_2012() -> RDDSample:
    """2012 annual round: Q = max annual DV 2011-2013 - 12.05."""
    return _round_sample(2012)


def load_1997_dvsite() -> RDDSample:
    """1997 round; Y = the change at the county's design-value (maximum) site."""
    return _round_sample(1997, "y_change_dvsite")


def _either(cuts):
    c = np.asarray(cuts, float)
    return lambda Q: (Q >= c[None, :] - EPS).any(axis=1).astype(int)


def load_two_score(y_col: str = "y_change") -> RDDSample:
    """1997 round, both standards: Q = (max annual DV, max 24-hour DV) in ug/m3, cutoffs
    (15.05, 65.5), D = 1 if either is at or above its cutoff. Counties need a valid DV
    for both standards."""
    cuts = (15.05, 65.5)
    df = _frame((1997,), y_col, need=("annual_dv", "daily_dv"))
    X, names = _design(df, (1997,))
    return RDDSample(
        Q=df[["annual_dv", "daily_dv"]].to_numpy(float), X=X, Y=df[y_col].to_numpy(float),
        threshold=cuts, name=f"epa_pm25_1997_two_score_{y_col}", feature_names=names,
        description=("PM2.5 nonattainment designations, 1997 NAAQS round. Q = (county max "
                     "annual DV, county max 24-hour DV), 2001-2003; treatment = 1{Q1 >= "
                     "15.05 or Q2 >= 65.5}; Y = change in annual mean PM2.5, 2001-2003 to "
                     "2008-2010."),
        citation=("EPA PM2.5 design value report 2006-2008 (2001-2003 site listing), EPA "
                  "Green Book (phistory), EPA AQS annual summaries; 70 FR 944"),
        treatment_rule=_either(cuts),
        extras={**_extras(df), "cutoffs": cuts, "rule": "either",
                "levels": (15.0, 65.0)},
    )


def load_pooled(y_col: str = "y_change") -> RDDSample:
    """All three rounds. Q = (max DV of the round's standard - cutoff) / level, a share of
    the standard; D = 1{Q >= 0}; round indicators in X; extras['county_id'] for
    clustering (counties repeat across rounds)."""
    df = _frame(tuple(ROUNDS), y_col)
    X, names = _design(df, tuple(ROUNDS))
    q = ((df.dv_main - df.cut_main) / df.level_main).to_numpy(float)
    return RDDSample(
        Q=q, X=X, Y=df[y_col].to_numpy(float), threshold=0.0,
        name=f"epa_pm25_pooled_{y_col}", feature_names=names,
        description=("PM2.5 nonattainment designations, 1997, 2006 and 2012 rounds pooled. "
                     "Q = (county max DV - cutoff) / NAAQS level; treatment = 1{Q >= 0}; "
                     "Y = change in annual mean PM2.5 from the DV period to 3-5 years after "
                     "designation."),
        citation=("EPA PM2.5 design value reports, Green Book, AQS; 70 FR 944, 74 FR 58688, "
                  "80 FR 2206"),
        treatment_rule=lambda Q: (Q >= -EPS).astype(int),
        extras={**_extras(df), "cutoff": 0.0, "level": df.level_main.to_numpy(float)},
    )
