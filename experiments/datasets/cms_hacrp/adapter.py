"""CMS Hospital-Acquired Condition (HAC) Reduction Program: a 1% Medicare payment cut for
the worst-scoring quartile of hospitals, FY 2015 to FY 2026.

Under section 1886(p) of the Social Security Act and 42 CFR 412.170-412.172, CMS scores
every subsection (d) hospital on hospital-acquired conditions and cuts all of its
Medicare FFS inpatient payments for the fiscal year by 1% if its Total HAC Score is in
the top (worst) quartile: "CMS will identify the 25 percent of hospitals with the highest
total scores" (412.172(e)(2)). The score, higher = worse:

    FY 2015-2017: points 1-10 by decile of each measure; Domain 1 = PSI 90 points,
                  Domain 2 = mean points of the CDC NHSN infection (HAI) measures
                  (CLABSI, CAUTI; + SSI from FY 2016; + MRSA, CDI from FY 2017);
                  Total = w1 D1 + w2 D2, (w1, w2) = (0.35, 0.65), (0.25, 0.75),
                  (0.15, 0.85).
    FY 2018-2019: the same domains on Winsorized z-scores, (w1, w2) = (0.15, 0.85).
    FY 2020-:     no domains; Total = mean of the Winsorized z-scores of the measures
                  the hospital has (PSI 90 and up to five HAIs).
    A hospital with only one domain (FY 2015-2019) gets that domain's score as its total.
    FY 2023: all measures suppressed (COVID-19), no Total HAC Scores, no payment cuts.

Penalized = 1{Total HAC Score > c_t}, c_t the 75th percentile of the year's scores among
non-Maryland hospitals. Maryland hospitals are scored but waived from the payment cut
(Maryland All-Payer Model) and left out of the percentile (CMS's FY 2016 file). Puerto
Rico hospitals and critical access hospitals are not subsection (d) hospitals and are not
in the program.

The cutoff used here is the one CMS deployed, read off the published payment-reduction
flags (FY 2016 CMS file, FY 2017- Hospital Compare): c_t is the midpoint between the
highest unpenalized and the lowest penalized non-Maryland score. Every year the flags are
an exact threshold on the published score (README). For FY 2015, which has no published
flag, c_t is the midpoint between the 75th percentile (7.0; ties at 7.0 not penalized)
and the next score up.

Q = load(), load_next_year(): (Total HAC Score - c_t) / sd_t, sd_t the standard
    deviation of the year's non-Maryland scores, so years on the decile-point scale
    (FY 2015-2017) and the z-score scale (FY 2018-) share one unit. D = 1{Q >= 0} =
    penalized (the worse side); no hospital has Q = 0. load_two_score(): Q = ((D1 -
    c_t)/sd_t, (D2 - c_t)/sd_t) with the year's domain weights, which sum to 1, so
    w'Q = (Total - c_t)/sd_t and D = 1{w'Q >= 0}. extras["q_raw"] = Total - c_t.
X = hospital characteristics: hospital ownership and emergency services (Hospital Compare
    general information file of that FY), state; log beds, log average daily census,
    occupancy, resident-to-bed ratio and teaching indicators, DSH patient percentage,
    Medicare day share, case-mix index, log Medicare discharges, urban / rural and
    provider type (SCH, MDH, RRC) from the FY's IPPS final rule Impact File; the previous
    program year's standardized Total HAC Score, penalty and standardized PSI 90 and HAI
    scores (missing indicators where absent); fiscal year indicators.
Y = load(): the standardized Total HAC Score, (score - c)/sd, of the first program year
    whose measurement windows start after the FY t result was known: FY t+3 (FY 2024 for
    FY 2020, as FY 2023 has no scores). Y > 0 means penalized again in that year.
    load_psi90(): that year's PSI 90 measure score, standardized within the year.
    load_next_year*(): the same in FY t+1, whose windows mostly precede the FY t result
    (a persistence / placebo outcome, not an effect of the penalty).

Sample: non-Maryland hospitals with a published Total HAC Score in a designation year and
Y observed. load(), load_psi90(): FY 2018-2022, the z-score years (FY 2015-2017 scores are
discrete, with large ties at the cutoff; load_all_years() adds them). load_next_year*():
FY 2018-2021, 2024, 2025. load_two_score(): FY 2018-2019, hospitals with both domains.
"""
from __future__ import annotations

import io
import os
import re
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample
from experiments.datasets.cms_hacrp.download import (CMS_HAC_FY2016, GENERAL, HAC_FILES,
                                                     IMPACT, RAW, general_path, hac_path,
                                                     impact_path)

HERE = Path(__file__).parent
PROCESSED = HERE / "data" / "processed"
CACHE = PROCESSED / "cms_hacrp_panel.csv.gz"

FYS = tuple(sorted(HAC_FILES))                    # 2015 ... 2026
SCORED_FYS = tuple(fy for fy in FYS if fy != 2023)  # FY 2023: suppressed, no scores
MEASURES = ("psi90", "clabsi", "cauti", "ssi", "mrsa", "cdi")
HAI = MEASURES[1:]
# Domain weights (Domain 1 = PSI 90, Domain 2 = HAI) where the score had two domains:
# 78 FR 50722 (FY 2015), 79 FR 50102-50104 (FY 2016), 80 FR 49575-49576 (FY 2017 on),
# confirmed for FY 2018 and FY 2019 in the impact analyses (82 FR 38568, 83 FR 41758).
# Equal measure weights from FY 2020: 83 FR 41486-41489.
WEIGHTS = {2015: (0.35, 0.65), 2016: (0.25, 0.75), 2017: (0.15, 0.85),
           2018: (0.15, 0.85), 2019: (0.15, 0.85)}
DECILE_FYS = (2015, 2016, 2017)                   # points 1-10; z-scores from FY 2018
# The main loaders use the z-score years: in FY 2015-2017 the score is discrete with large
# ties at the cutoff (README), which breaks continuity at the cutoff.
MAIN_YEARS = (2018, 2019, 2020, 2021, 2022)
# Measurement windows (PSI 90; HAI) of each program year, from the HACRP files.
WINDOWS = {
    2015: ("2011-07-01", "2013-06-30", "2012-01-01", "2013-12-31"),
    2016: ("2012-07-01", "2014-06-30", "2013-01-01", "2014-12-31"),
    2017: ("2013-07-01", "2015-06-30", "2014-01-01", "2015-12-31"),
    2018: ("2014-07-01", "2015-09-30", "2015-01-01", "2016-12-31"),
    2019: ("2015-10-01", "2017-06-30", "2016-01-01", "2017-12-31"),
    2020: ("2016-07-01", "2018-06-30", "2017-01-01", "2018-12-31"),
    2021: ("2017-07-01", "2019-06-30", "2018-01-01", "2019-12-31"),
    2022: ("2018-07-01", "2019-12-31", "2019-01-01", "2019-12-31"),
    2023: (None, None, "2021-01-01", "2021-12-31"),
    2024: ("2021-01-01", "2022-06-30", "2022-01-01", "2022-12-31"),
    2025: ("2021-07-01", "2023-06-30", "2022-01-01", "2023-12-31"),
    2026: ("2022-07-01", "2024-06-30", "2023-01-01", "2024-12-31"),
}
# Designation year -> first program year whose PSI 90 window starts on or after July 1
# of t-1 and whose HAI window starts on or after January 1 of t (the FY t result reaches
# hospitals around the start of FY t, October 1 of t-1). FY 2023 has no scores, so FY 2020
# maps to FY 2024.
OUTCOME_FY = {2015: 2018, 2016: 2019, 2017: 2020, 2018: 2021, 2019: 2022, 2020: 2024,
              2021: 2024, 2022: 2025}
NEXT_FY = {2015: 2016, 2016: 2017, 2017: 2018, 2018: 2019, 2019: 2020, 2020: 2021,
           2021: 2022, 2024: 2025, 2025: 2026}
TWO_SCORE_YEARS = (2018, 2019)       # z-score years with weights (0.15, 0.85); FY 2017
                                     # has the same weights on decile points
EPS = 1e-9

_NA = {"", "nan", "n/a", "not available", "na", "none"}


def _require(*paths: Path) -> None:
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(
                f"{p} missing. Run python -m experiments.datasets.cms_hacrp.download")


def _num(s: pd.Series) -> pd.Series:
    t = s.astype(str).str.replace("*", "", regex=False).str.replace(",", "", regex=False)
    t = t.str.strip()
    return pd.to_numeric(t.where(~t.str.lower().isin(_NA)), errors="coerce")


def _ccn(s: pd.Series) -> pd.Series:
    t = s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    return t.str.zfill(6)


# ----------------------------------------------------------------------- HACRP files

def _pick(cols: list[str], pattern: str) -> str | None:
    hits = [c for c in cols if re.fullmatch(pattern, c)]
    return hits[0] if hits else None


def _read_hac(fy: int) -> pd.DataFrame:
    """One program year in a common layout: ccn, name, state, total, d1, d2, the six
    measure scores (decile points or Winsorized z-scores), the published payment
    reduction flag and the total-score footnote."""
    raw = pd.read_csv(hac_path(fy), dtype=str, keep_default_na=False, encoding="latin-1")
    raw.columns = [c.strip().lower().replace(" ", "_") for c in raw.columns]
    cols = list(raw.columns)
    out = pd.DataFrame({"fy": fy}, index=raw.index)
    out["ccn"] = _ccn(raw[_pick(cols, r"provider_?id|facility_id")])
    out["name"] = raw[_pick(cols, r"hospital_name|facility_name")].str.strip()
    out["state"] = raw["state"].str.strip()
    out["total"] = _num(raw["total_hac_score"])
    for d in ("d1", "d2"):
        c = _pick(cols, rf"domain_{d[1]}_score")
        out[d] = _num(raw[c]) if c else np.nan
    for m in MEASURES:
        pat = r"(ahrq_)?psi_?90(_w_z)?_score" if m == "psi90" else rf"{m}(_w_z)?_score"
        c = _pick(cols, pat)
        out[m] = _num(raw[c]) if c else np.nan
    pay = raw.get("payment_reduction", pd.Series("", index=raw.index)).str.strip()
    out["pen_pub"] = pay.map({"Yes": 1.0, "No": 0.0})
    fn = _pick(cols, r"total_hac(_score)?_footnote|footnotes")
    out["total_fn"] = raw[fn].str.replace(r"\.0$", "", regex=True).str.strip() if fn else ""
    out.loc[out.total_fn.str.lower().isin(_NA), "total_fn"] = ""
    return out.drop_duplicates("ccn").reset_index(drop=True)


def _read_cms_fy2016() -> pd.DataFrame:
    """CMS's FY 2016 HAC Reduction Program Information (updated February 2016): Total HAC
    Score and the published "Top 25th Percentile" flag. Its note: the 75th percentile is
    6.7500 and Maryland hospitals are not included in determining the top quartile."""
    with zipfile.ZipFile(RAW / CMS_HAC_FY2016) as z:
        body = z.read("HAC_Results_FY16_Feb 2016.txt")
    c = pd.read_csv(io.BytesIO(body), sep="\t", skiprows=1, dtype=str, keep_default_na=False,
                    encoding="latin-1").iloc[:, :4]
    c.columns = ["ccn", "state", "top25", "total"]
    c = c[c.ccn.str.fullmatch(r"\d{6}")]
    return pd.DataFrame({"ccn": c.ccn, "total_cms": _num(c.total),
                         "pen_cms": c.top25.map({"Y": 1.0, "N": 0.0})}).reset_index(drop=True)


def _recomputed_total(df: pd.DataFrame, fy: int) -> pd.Series:
    """Total HAC Score from the published domain or measure scores."""
    if fy in WEIGHTS:
        w1, w2 = WEIGHTS[fy]
        d1, d2 = df.d1, df.d2
        return pd.Series(np.where(d1.notna() & d2.notna(), w1 * d1 + w2 * d2,
                                  np.where(d1.notna(), d1, d2)), index=df.index)
    return df[list(MEASURES)].mean(axis=1, skipna=True)


def _hac_panel() -> pd.DataFrame:
    frames = []
    cms16 = _read_cms_fy2016()
    for fy in FYS:
        d = _read_hac(fy)
        d["total_compare"] = d.total
        if fy == 2016:   # the CMS file is the official FY 2016 score and flag
            d = d.merge(cms16, on="ccn", how="outer")
            d["fy"] = 2016
            d["total"] = d.total_cms
            d["pen_pub"] = d.pen_cms
            d = d.drop(columns=["total_cms", "pen_cms"])
        d["total_recomputed"] = _recomputed_total(d, fy)
        d["n_hai"] = d[list(HAI)].notna().sum(axis=1).astype(float)
        frames.append(d)
    p = pd.concat(frames, ignore_index=True)
    p["maryland"] = (p.state == "MD") | p.ccn.str.startswith("21")
    return p


# ------------------------------------------------------------------------------- rule

def year_parameters(p: pd.DataFrame) -> pd.DataFrame:
    """Per program year, among non-Maryland hospitals with a Total HAC Score: the 75th
    percentile, the deployed cutoff read off the published flags, mean and sd."""
    rows = []
    for fy in SCORED_FYS:
        d = p[(p.fy == fy) & ~p.maryland & p.total.notna()]
        s = d.total
        q75 = float(np.percentile(s, 75))
        if d.pen_pub.notna().any():
            c_low = float(s[d.pen_pub == 0].max())
            c_high = float(s[d.pen_pub == 1].min())
            source = "published flag"
        else:                                           # FY 2015: the 75th percentile
            c_low, c_high = q75, float(s[s > q75 + EPS].min())
            source = "75th percentile"
        rows.append({"fy": fy, "n_scored": len(s), "mean": float(s.mean()),
                     "sd": float(s.std()), "q75": q75, "c_low": c_low, "c_high": c_high,
                     "cutoff": (c_low + c_high) / 2, "cutoff_source": source,
                     "share_not_penalized": float((s < c_high).mean())})
    return pd.DataFrame(rows).set_index("fy")


# ------------------------------------------------------------------------- covariates

def _read_general(fy: int) -> pd.DataFrame:
    g = pd.read_csv(general_path(fy), dtype=str, keep_default_na=False, encoding="latin-1")
    g.columns = [c.strip().lower() for c in g.columns]
    cols = list(g.columns)
    own = g[_pick(cols, r"hospital_?ownership")].str.strip().str.lower()
    grp = np.select([own.str.startswith("government"), own.str.startswith("proprietary"),
                     own.str.contains("church"), own.str.startswith("voluntary"),
                     own.str.startswith("physician")],
                    ["government", "proprietary", "nonprofit_church", "nonprofit",
                     "physician"], "other")
    em = g[_pick(cols, r"emergency_?services?")].str.strip().str.lower()
    return pd.DataFrame({
        "ccn": _ccn(g[_pick(cols, r"provider_?id|facility_id")]),
        "hospital_type": g[_pick(cols, r"hospital_?type")].str.strip(),
        "ownership": grp,
        "emergency": em.isin(["yes", "true", "y"]).astype(float).where(em != ""),
    }).drop_duplicates("ccn")


def _read_impact(fy: int) -> pd.DataFrame:
    with zipfile.ZipFile(impact_path(fy)) as z:
        names = [n for n in z.namelist() if n.endswith(".txt") and "ariable" not in n]
        fr = [n for n in names if re.search(r"Final Rule|FR data|FR Impact", n)
              and not re.search(r"Correct|CN data|\bCN\b", n)]
        body = z.read(fr[0]).decode("latin-1")
    lines = body.splitlines()
    hi = next(i for i, l in enumerate(lines[:10]) if "Provider Number" in l)
    d = pd.read_csv(io.StringIO(body), sep="\t", skiprows=hi, dtype=str,
                    keep_default_na=False)
    d.columns = [re.sub(r"\s+", " ", c.strip().strip('"')) for c in d.columns]
    cols = list(d.columns)

    def col(*pats):
        for pat in pats:
            c = next((c for c in cols if re.fullmatch(pat, c, flags=re.I)), None)
            if c:
                return c
        raise KeyError(f"FY {fy} impact file: none of {pats}")
    ptype = col(r"Provider Type", r"Provider Type for first half.*")
    cmi = next(c for c in cols if re.fullmatch(r"TACMIV\d+", c))
    out = pd.DataFrame({
        "ccn": _ccn(d[col(r"Provider Number")]),
        "beds": _num(d[col(r"Beds")]),
        "adc": _num(d[col(r"Average Daily Census")]),
        "rb_ratio": _num(d[col(r"Resident to Bed Ratio")]),
        "dsh_pct": _num(d[col(r"DSHPCT")]),
        "mcr_pct": _num(d[col(r"MCR_PCT", r"Medicare PCT", r"Medicare Percentage")]),
        "cmi": _num(d[cmi]),
        "bills": _num(d[col(r"Bills")]),
        "urgeo": d[col(r"URGEO")].str.strip().str.upper(),
        "ptype": _num(d[ptype]),
    })
    return out.drop_duplicates("ccn")


# ------------------------------------------------------------------------------ build

def build() -> pd.DataFrame:
    """Hospital x program year panel FY 2015-2026: scores, the rule, Impact File and
    general-information characteristics."""
    if not CACHE.exists():
        df = _build()
        PROCESSED.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=PROCESSED, prefix=".tmp_", suffix=".csv.gz")
        os.close(fd)
        try:
            df.to_csv(tmp, index=False, compression="gzip")
            os.chmod(tmp, 0o644)
            os.replace(tmp, CACHE)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    p = pd.read_csv(CACHE, keep_default_na=False, na_values=[""], low_memory=False,
                    dtype={"ccn": str, "total_fn": str, "urgeo": str, "ownership": str,
                           "hospital_type": str, "state": str, "name": str})
    for c in ["total_fn", "urgeo", "ownership", "hospital_type", "state", "name"]:
        p[c] = p[c].fillna("").astype(str)
    p["maryland"] = p.maryland.astype(bool)
    return p


def _build() -> pd.DataFrame:
    _require(*[hac_path(fy) for fy in FYS], RAW / CMS_HAC_FY2016,
             *[general_path(fy) for fy in GENERAL], *[impact_path(fy) for fy in IMPACT])
    p = _hac_panel()
    par = year_parameters(p)
    p = p.merge(par[["cutoff", "sd", "mean", "q75"]].add_prefix("yr_"),
                left_on="fy", right_index=True, how="left")
    scored = p.total.notna() & ~p.maryland & p.fy.isin(SCORED_FYS)
    p["q"] = np.where(scored, (p.total - p.yr_cutoff) / p.yr_sd, np.nan)
    p["pen_rule"] = np.where(scored, (p.q >= 0).astype(float), np.nan)
    p["pen_q75"] = np.where(scored, (p.total > p.yr_q75 + EPS).astype(float), np.nan)
    # Measure scores standardized within the year (non-Maryland hospitals).
    nm = ~p.maryland
    for m in ("psi90", "d1", "d2"):
        g = p[nm].groupby("fy")[m]
        p[f"{m}_std"] = (p[m] - p.fy.map(g.mean())) / p.fy.map(g.std())
    hai = p[list(HAI)].mean(axis=1, skipna=True)
    g = hai[nm].groupby(p.fy[nm])
    p["hai_std"] = (hai - p.fy.map(g.mean())) / p.fy.map(g.std())
    gen = pd.concat([_read_general(fy).assign(fy=fy) for fy in GENERAL])
    imp = pd.concat([_read_impact(fy).assign(fy=fy) for fy in IMPACT])
    p = p.merge(gen, on=["ccn", "fy"], how="left").merge(imp, on=["ccn", "fy"], how="left")
    for c in ["hospital_type", "ownership", "urgeo"]:
        p[c] = p[c].fillna("")
    return p.sort_values(["fy", "ccn"]).reset_index(drop=True)


# ----------------------------------------------------------------------------- checks

def rule_check() -> dict:
    """The deployed rule against the published flags, per program year, and the
    recomputed Total HAC Score against the published one."""
    p = build()
    par = year_parameters(p)
    out = {}
    for fy in SCORED_FYS:
        d = p[(p.fy == fy) & ~p.maryland]
        s = d[d.total.notna()]
        pub = s.pen_pub
        has = pub.notna()
        rec_ok = (np.abs(s.total_recomputed - s.total) <= 2e-4)
        r = {
            "non_md_scored": int(len(s)),
            "penalized_rule": int(s.pen_rule.sum()),
            "penalized_published": int((pub == 1).sum()) if has.any() else None,
            "cutoff_interval": (par.at[fy, "c_low"], par.at[fy, "c_high"]),
            "q75": par.at[fy, "q75"],
            "share_not_penalized": round(par.at[fy, "share_not_penalized"], 4),
            "rule_agree_published": float((s.pen_rule == pub)[has].mean()) if has.any() else None,
            "q75_rule_agree_published": float((s.pen_q75 == pub)[has].mean()) if has.any() else None,
            "q75_rule_mismatch": int((s.pen_q75 != pub)[has].sum()) if has.any() else None,
            "flag_without_score": d[d.total.isna()].pen_pub.value_counts().to_dict(),
            "recomputed_total_within_2e-4": float(rec_ok.mean()),
            "maryland_scored": int(p[(p.fy == fy) & p.maryland].total.notna().sum()),
            "maryland_flags": p[(p.fy == fy) & p.maryland].pen_pub.value_counts().to_dict(),
        }
        out[fy] = r
    d16 = p[(p.fy == 2016) & ~p.maryland]
    chg = d16[(d16.total.isna() != d16.total_compare.isna())
              | (np.abs(d16.total - d16.total_compare) > 1e-9)]
    out["fy2016_cms_vs_compare"] = chg[["ccn", "total", "total_compare", "pen_pub"]].to_dict(
        "records")
    out["fy2023"] = {"scored": int(p[p.fy == 2023].total.notna().sum()),
                     "flags": p[p.fy == 2023].pen_pub.value_counts().to_dict()}
    return out


# ---------------------------------------------------------------------------- samples

def _lagged(p: pd.DataFrame) -> pd.DataFrame:
    """Previous scored program year (FY 2022 for FY 2024) of each hospital."""
    prev = {fy: max([f for f in SCORED_FYS if f < fy], default=None) for fy in FYS}
    lag = p[["ccn", "fy", "q", "pen_rule", "psi90_std", "hai_std"]].copy()
    lag["fy"] = lag.fy.map({v: k for k, v in prev.items() if v is not None})
    lag = lag.dropna(subset=["fy"]).astype({"fy": int})
    return lag.rename(columns={"q": "lag_q", "pen_rule": "lag_pen", "psi90_std": "lag_psi90",
                               "hai_std": "lag_hai"})


def _frame(outcome: dict[int, int], y_col: str) -> pd.DataFrame:
    p = build()
    base = p[p.fy.isin(list(outcome)) & ~p.maryland & p.q.notna()].copy()
    fut = p[~p.maryland & p.fy.isin(list(outcome.values()))][
        ["ccn", "fy", "q", "pen_rule", "psi90_std", "total", "psi90"]]
    fut = fut.rename(columns={"fy": "y_fy", "q": "y_total", "pen_rule": "y_penalized",
                              "psi90_std": "y_psi90", "total": "y_total_raw",
                              "psi90": "y_psi90_raw"})
    base["y_fy"] = base.fy.map(outcome)
    base = base.merge(fut, on=["ccn", "y_fy"], how="left")
    base = base.merge(_lagged(p), on=["ccn", "fy"], how="left")
    base = base[base[y_col].notna()]
    return base.sort_values(["fy", "ccn"]).reset_index(drop=True)


def _design(df: pd.DataFrame) -> tuple:
    X = pd.DataFrame(index=df.index)
    for c, f in [("beds", np.log), ("adc", np.log1p), ("bills", np.log1p)]:
        v = df[c].where(df[c] > 0)
        X[f"log_{c}"] = f(v).fillna(f(v).median())
    X["occupancy"] = (df.adc / df.beds.where(df.beds > 0)).clip(0, 1.5)
    X["occupancy"] = X.occupancy.fillna(X.occupancy.median())
    X["rb_ratio"] = df.rb_ratio.fillna(0.0)
    X["teaching"] = (df.rb_ratio > 0).astype(float)
    X["major_teaching"] = (df.rb_ratio >= 0.25).astype(float)
    for c in ["dsh_pct", "mcr_pct", "cmi"]:
        X[c] = df[c].fillna(df[c].median())
    X["impact_missing"] = df.beds.isna().astype(float)
    X["urban_other"] = (df.urgeo == "OURBAN").astype(float)      # large urban base
    X["rural"] = (df.urgeo == "RURAL").astype(float)
    pt = df.ptype
    X["sch"] = pt.isin([16, 17]).astype(float)
    X["mdh"] = pt.isin([14, 15]).astype(float)
    X["rrc"] = pt.isin([7, 15, 17, 22]).astype(float)
    for o in ["government", "proprietary", "nonprofit_church", "physician", "other"]:
        X[f"own_{o}"] = (df.ownership == o).astype(float)      # voluntary nonprofit base
    X["own_missing"] = (df.ownership == "").astype(float)
    X["emergency"] = df.emergency.fillna(1.0)
    X["emergency_missing"] = df.emergency.isna().astype(float)
    X["lag_q"] = df.lag_q.fillna(0.0)
    X["lag_pen"] = df.lag_pen.fillna(0.0)
    X["lag_missing"] = df.lag_q.isna().astype(float)
    for c in ["lag_psi90", "lag_hai"]:
        X[c] = df[c].fillna(0.0)
        X[f"{c}_missing"] = df[c].isna().astype(float)
    for fy in sorted(df.fy.unique())[1:]:
        X[f"fy{fy}"] = (df.fy == fy).astype(float)
    sv = df.state.value_counts()
    states = sorted(sv[sv >= 30].index)
    for s in states:
        if s != "CA":                                           # California base
            X[f"state_{s}"] = (df.state == s).astype(float)
    X["state_other"] = (~df.state.isin(states)).astype(float)
    X = X.loc[:, X.std() > 0]
    return X.to_numpy(float), list(X.columns)


def _extras(df: pd.DataFrame) -> dict:
    return {
        "ccn": df.ccn.to_numpy(str), "fy": df.fy.to_numpy(int), "name": df.name.to_numpy(str),
        "state": df.state.to_numpy(str), "y_fy": df.y_fy.to_numpy(int),
        "total_hac_score": df.total.to_numpy(float),
        "cutoff_raw": df.yr_cutoff.to_numpy(float), "sd_raw": df.yr_sd.to_numpy(float),
        "q75_raw": df.yr_q75.to_numpy(float),
        "q_raw": (df.total - df.yr_cutoff).to_numpy(float),
        "eligible_published": df.pen_pub.to_numpy(float),
        "penalized_q75_rule": df.pen_q75.to_numpy(float),
        "domain1": df.d1.to_numpy(float), "domain2": df.d2.to_numpy(float),
        **{m: df[m].to_numpy(float) for m in MEASURES},
        "n_hai_measures": df.n_hai.to_numpy(float),
        "y_total": df.y_total.to_numpy(float), "y_penalized": df.y_penalized.to_numpy(float),
        "y_psi90": df.y_psi90.to_numpy(float), "y_total_raw": df.y_total_raw.to_numpy(float),
        "y_psi90_raw": df.y_psi90_raw.to_numpy(float),
        "lag_q": df.lag_q.to_numpy(float), "lag_penalized": df.lag_pen.to_numpy(float),
        "beds": df.beds.to_numpy(float), "rb_ratio": df.rb_ratio.to_numpy(float),
        "decile_scale": df.fy.isin(DECILE_FYS).to_numpy(float),
        "treated_side": "penalized: Q >= 0 (Total HAC Score above the cutoff; higher = worse)",
    }


_CITATION = ("CMS Hospital-Acquired Condition Reduction Program, section 1886(p) of the "
             "Social Security Act, 42 CFR 412.170-412.172; HACRP hospital files FY 2015-2026 "
             "(Hospital Compare archives via data.nber.org/compare/hospital); CMS FY 2016 HAC "
             "Reduction Program Information; IPPS final rule Impact Files FY 2015-2025 "
             "(cms.gov)")
_Y_LABEL = {"y_total": "standardized Total HAC Score, (score - cutoff)/sd, in FY {h}",
            "y_psi90": "PSI 90 measure score standardized within FY {h}"}


def _single(outcome: dict[int, int], y_col: str, tag: str, horizon: str) -> RDDSample:
    df = _frame(outcome, y_col)
    X, names = _design(df)
    ex = _extras(df)
    ex.update({"cutoff": 0.0, "outcome_fy": outcome, "horizon": horizon})
    return RDDSample(
        Q=df.q.to_numpy(float), X=X, Y=df[y_col].to_numpy(float), threshold=0.0,
        name=f"cms_hacrp_{tag}", feature_names=names,
        description=("CMS HAC Reduction Program, non-Maryland subsection (d) hospitals, "
                     f"program years {min(outcome)}-{max(outcome)}. Q = (Total HAC Score - "
                     "deployed 75th-percentile cutoff) / sd of the year's scores; treatment "
                     "= 1{Q >= 0}, a 1% Medicare payment cut; Y = "
                     + _Y_LABEL[y_col].format(h=horizon) + "."),
        citation=_CITATION, treatment_rule=lambda Q: (Q >= 0).astype(int), extras=ex,
    )


def _years(outcome: dict[int, int], years) -> dict[int, int]:
    return {fy: y for fy, y in outcome.items() if fy in years}


def load() -> RDDSample:
    """Q = standardized Total HAC Score minus cutoff, FY 2018-2022 (z-score years);
    D = penalized; Y = standardized Total HAC Score in FY t+3 (FY 2024 for FY 2020)."""
    return _single(_years(OUTCOME_FY, MAIN_YEARS), "y_total", "total", "t+3")


def load_psi90() -> RDDSample:
    """As ``load()``, Y = standardized PSI 90 score in FY t+3 (FY 2024 for FY 2020)."""
    return _single(_years(OUTCOME_FY, MAIN_YEARS), "y_psi90", "psi90", "t+3")


def load_all_years() -> RDDSample:
    """As ``load()`` with FY 2015-2017 added (decile-point scores, discrete Q with ties at
    the cutoff; see README)."""
    return _single(OUTCOME_FY, "y_total", "all_years_total", "t+3")


def load_next_year() -> RDDSample:
    """Y = standardized Total HAC Score in FY t+1 (FY 2018-2021, 2024, 2025). Its windows
    mostly precede the FY t result: persistence, not an effect of the penalty."""
    return _single(_years(NEXT_FY, range(2018, 2027)), "y_total", "next_year_total", "t+1")


def load_next_year_psi90() -> RDDSample:
    return _single(_years(NEXT_FY, range(2018, 2027)), "y_psi90", "next_year_psi90", "t+1")


def load_two_score(years: tuple = TWO_SCORE_YEARS, y_col: str = "y_total") -> RDDSample:
    """Q = ((D1 - c)/sd, (D2 - c)/sd) for hospitals with both domains, D = 1{w'Q >= 0},
    w = the year's domain weights (one weight vector per call). D2 is the value implied
    by the published total and D1 ((total - w1 D1)/w2, within rounding of the published
    D2), so w'Q reproduces the published total exactly."""
    ws = {WEIGHTS[fy] for fy in years}
    if len(ws) != 1:
        raise ValueError(f"years {years} have different domain weights {ws}")
    w = np.array(ws.pop())
    out = {fy: OUTCOME_FY[fy] for fy in years}
    df = _frame(out, y_col)
    df = df[df.d1.notna() & df.d2.notna()].reset_index(drop=True)
    d2_impl = (df.total - w[0] * df.d1) / w[1]
    Q = np.column_stack([(df.d1 - df.yr_cutoff) / df.yr_sd,
                         (d2_impl - df.yr_cutoff) / df.yr_sd])
    X, names = _design(df)
    ex = _extras(df)
    ex.update({"weights": tuple(w), "cutoff": 0.0, "outcome_fy": out, "horizon": "t+3",
               "domain2_implied": d2_impl.to_numpy(float),
               "treated_side": (f"penalized: {w[0]:g} Q1 + {w[1]:g} Q2 >= 0 (higher = worse); "
                                "Q1 = PSI 90 (Domain 1), Q2 = HAI mean (Domain 2)")})
    return RDDSample(
        Q=Q, X=X, Y=df[y_col].to_numpy(float), threshold=(0.0, 0.0),
        name=f"cms_hacrp_two_score_{'_'.join(map(str, years))}_{y_col}", feature_names=names,
        description=(f"CMS HAC Reduction Program, FY {', '.join(map(str, years))}, hospitals "
                     "with both domains. Q = (Domain 1 (PSI 90), Domain 2 (HAI)) minus the "
                     "year's cutoff, over the sd of the year's Total HAC Scores; treatment = "
                     f"1{{{w[0]:g} Q1 + {w[1]:g} Q2 >= 0}}, a 1% payment cut; Y = "
                     + _Y_LABEL[y_col].format(h="t+3") + "."),
        citation=_CITATION, treatment_rule=lambda Q: (Q @ w >= -1e-12).astype(int), extras=ex,
    )


def load_two_score_psi90() -> RDDSample:
    return load_two_score(TWO_SCORE_YEARS, "y_psi90")
