"""Medicare ESRD Quality Incentive Program (ESRD QIP): dialysis facilities' Total
Performance Score (TPS), a weighted sum of measure scores, with payment-reduction bands.

Under 42 CFR 413.177-413.178 (section 1881(h) of the Social Security Act) CMS scores every
outpatient dialysis facility (CCN) each payment year (PY). Each measure gets 0-10 points
(the better of achievement and improvement against national standards); measures are
grouped in domains with weights set in each year's ESRD PPS final rule, and

    TPS = round_half_up(10 * sum_i w_i * s_i)                 (0-100 from PY2014)

with w_i the measure weights after redistributing the weight of measures (and domains) a
facility has no score on. A facility whose TPS is below the year's minimum TPS m loses
0.5% of its Medicare ESRD payments for every 10 points (or part) below m, up to 2%:

    reduction = 0                                   if TPS >= m
                min(2, 0.5 * ceil((m - TPS) / 10))  otherwise,

so the cut points on the TPS are m, m - 10, m - 20, m - 30 (PY2014 on; PY2012-2013 used a
0-30 scale, README). The reduction applies to every Medicare payment in calendar year PY,
and the facility must post its score certificate during PY. The performance period is
calendar year PY - 2, and final scores are published in December of PY - 1.

Data: NBER's mirror of the Dialysis Facility Compare download (the CMS site blocks
scripts) carries the "Complete QIP Data" of PY2013, PY2014 (truncated, README), PY2015,
PY2017, PY2018 and PY2019, and the DFC facility file of each release.

Q = load(), load_bands(): TPS - m, the published TPS minus the year's minimum TPS.
    load_two_score(): (clinical domain score, reporting domain score), each a weighted
    mean of its measure scores on 0-100 with the year's within-domain weights; the TPS is
    0.75 Q1 + 0.25 Q2 in both pooled years (PY2015, PY2017).
D = load(): 1{Q >= 0}, no payment reduction. load_bands(): the number of the four cut
    points (0, -10, -20, -30) at or below Q, 0-4; the reduction is 0.5 * (4 - D) %.
    load_two_score(): 1{0.75 Q1 + 0.25 Q2 >= m - 0.5}, m = 60.
X = facility characteristics at the designation (DFC release after the final scores):
    for-profit, chain (DaVita, Fresenius, other chain; independent base), log dialysis
    stations, late shift (after 5 pm), in-center PD and home HD training offered, years
    since CMS certification, ESRD Network (18 regions); and lagged values from the QIP
    file's improvement period (calendar year PY - 3, before the performance period): % of
    patient-months with a fistula, with a long-term catheter, with adult HD Kt/V >= 1.2,
    log adult HD patient-months, PD share of patient-months; the TPS of PY - 2 (z-scored
    within PY, performance period PY - 4). Payment-year indicators.
Y = load(), load_bands(), load_two_score(): the TPS of PY + 2, whose performance period
    (calendar year PY) is the year the reduction applies, after the final scores were
    published. load_clinical(): the clinical domain score of PY + 2. load_next_py(): the
    TPS of PY + 1 (performance period PY - 1, which ends before the final PY scores are
    published, so it is mostly pre-treatment; PY2014, PY2017, PY2018).
"""
from __future__ import annotations

import gzip
import io
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample
from experiments.datasets.esrd_qip.download import QIP_ZIP, RAW

HERE = Path(__file__).parent
PROCESSED = HERE / "data" / "processed"
CACHE = PROCESSED / "esrd_qip_facility_years.csv.gz"

PYS = (2013, 2014, 2015, 2017, 2018, 2019)
# Minimum TPS (Federal Register; README). PY2014: the rule (76 FR 70282) only estimated
# 56 and said the final value would be posted on dialysisreports.org; the published
# reductions imply 53 exactly (README).
MIN_TPS = {2013: 30, 2014: 53, 2015: 60, 2017: 60, 2018: 49, 2019: 60}
BAND = 10.0                       # TPS points per 0.5% step (PY2014 on)
MAIN_PYS = (2015, 2017)           # TPS of PY + 2 in the mirror (PY2017, PY2019)
NEXT_PYS = (2014, 2017, 2018)     # TPS of PY + 1 in the mirror
TWO_SCORE_WEIGHTS = (0.75, 0.25)  # clinical, reporting: PY2015 and PY2017
WEIGHT_DIGITS = 5                 # CMS's per-measure weights carry finite precision
EPS = 1e-9

# Domains: name -> (weight in the TPS, {measure-score column: relative weight}).
# "prop": a missing measure's weight is shared in proportion to the others in its domain
# (PY2014-PY2017, equal weights but hypercalcemia at 2/3 in PY2017); "even": shared
# equally among the other measures in its domain, and a missing domain's weight equally
# among all scored measures (PY2018-PY2019). README gives the citations.
SPEC: dict[int, dict] = {
    2014: {"redistribute": "prop", "domains": {
        "clinical": (0.90, {"Hemoglobin 12 g dL Measure Score": 1, "URR Measure Score": 1,
                            "Vascular Access Combined Measure Score": 1}),
        "reporting": (0.10, {"ICH CAHPS Admin Score": 1, "NHSN Event Reporting Score": 1,
                             "Mineral Metabolism Reporting Score": 1})}},
    2015: {"redistribute": "prop", "domains": {
        "clinical": (0.75, {"Hemoglobin 12 g/dL Measure Score": 1,
                            "Kt/V Dialysis Adequacy Combined Measure Score": 1,
                            "Vascular Access Combined Measure Score": 1}),
        "reporting": (0.25, {"ICH CAHPS Admin Score": 1, "NHSN Event Reporting Score": 1,
                             "Mineral Metabolism Reporting Score": 1,
                             "Anemia Management Reporting Score": 1})}},
    2017: {"redistribute": "prop", "domains": {
        "clinical": (0.75, {"VAT Combined Measure Score": 1,
                            "Kt/V Dialysis Adequacy Combined Measure Score": 1,
                            "Hypercalcemia Measure Score": 2 / 3, "NHSN Measure Score": 1,
                            "SRR Measure Score": 1}),
        "reporting": (0.25, {"ICH CAHPS Admin Score": 1,
                             "Mineral Metabolism Reporting Score": 1,
                             "Anemia Management Reporting Score": 1})}},
    2018: {"redistribute": "even", "domains": {
        "clinical": (0.90, {"NHSN BSI Measure Score": 20, "ICH CAHPS Measure Score": 20,
                            "SRR Measure Score": 10,
                            "Kt/V Dialysis Adequacy Combined Measure Score": 18,
                            "Vascular Access Combined Measure Score": 18,
                            "STrR Measure Score": 7, "Hypercalcemia Measure Score": 7}),
        "reporting": (0.10, {"Mineral Metabolism Measure Score": 1,
                             "Anemia Management Measure Score": 1,
                             "Clinical Depression Screening and Follow-up Measure Score": 1,
                             "Pain Assessment and Follow-up Measure Score": 1,
                             "NHSN Influenza Measure Score": 1})}},
    2019: {"redistribute": "even", "domains": {
        "clinical": (0.75, {"ICH CAHPS Measure Score": 26,
                            "Standardized Readmission Ratio (SRR) Measure Score": 16,
                            "Standardized Transfusion Ratio (STrR) Measure Score": 12,
                            "Kt/V Comprehensive Measure Score": 19,
                            "Vascular Access Combined Measure Score": 19,
                            "Hypercalcemia Measure Score": 8}),
        "safety": (0.15, {"NHSN Combined Measure Score": 1}),
        "reporting": (0.10, {"Mineral Metabolism Measure Score": 1,
                             "Anemia Management Measure Score": 1,
                             "Clinical Depression Screening and Follow-up Measure Score": 1,
                             "Pain Assessment and Follow-up Measure Score": 1,
                             "NHSN Influenza Measure Score": 1})}},
}

# Lagged (improvement / baseline period, calendar year PY - 3) rates and patient-months.
# Each entry: (numerator column, denominator column) or a column of rates in %.
_LAG = {
    2014: {"fistula": "Fistula Baseline Period Rate", "catheter": "Catheter Baseline Period Rate",
           "hd_pm": "Number of Fistula Patient Months Baseline Period"},
    2015: {"fistula": "Fistula Baseline Period Rate", "catheter": "Catheter Baseline Period Rate",
           "ktv_hd": "Kt/V Adult Hemodialysis Baseline Period Rate",
           "hd_pm": "Number of Kt/V Adult Hemodialysis Patient Months Baseline Period",
           "pd_pm": "Number of Kt/V Adult Peritoneal Dialysis Patient Months Baseline Period"},
    2017: {"fistula": ("VAT Fistula Improvement Period Numerator",
                       "VAT Fistula Improvement Period Denominator"),
           "catheter": ("VAT Catheter Improvement Period Numerator",
                        "VAT Catheter Improvement Period Denominator"),
           "ktv_hd": ("Kt/V Adult Hemodialysis Improvement Period Numerator",
                      "Kt/V Adult Hemodialysis Improvement Period Denominator"),
           "hd_pm": "Kt/V Adult Hemodialysis Improvement Period Denominator",
           "pd_pm": "Kt/V Adult Peritoneal Dialysis Improvement Period Denominator"},
    2018: {"fistula": ("VAT Fistula Improvement Period Numerator",
                       "VAT Fistula Improvement Period Denominator"),
           "catheter": ("VAT Catheter Improvement Period Numerator",
                        "VAT Catheter Improvement Period Denominator"),
           "ktv_hd": ("Kt/V Adult Hemodialysis Improvement Period Numerator",
                      "Kt/V Adult Hemodialysis Improvement Period Denominator"),
           "hd_pm": "Kt/V Adult Hemodialysis Improvement Period Denominator",
           "pd_pm": "Kt/V Adult Peritoneal Dialysis Improvement Period Denominator"},
    2019: {"fistula": ("VAT Fistula Improvement Period Numerator",
                       "VAT Fistula Improvement Period Denominator"),
           "catheter": ("VAT Catheter Improvement Period Numerator",
                        "VAT Catheter Improvement Period Denominator")},
}


def _require() -> None:
    for py in PYS:
        p = RAW / QIP_ZIP[py]
        if not p.exists():
            raise FileNotFoundError(
                f"{p} missing. Run python -m experiments.datasets.esrd_qip.download")


def _zip(py: int) -> zipfile.ZipFile:
    b = (RAW / QIP_ZIP[py]).read_bytes()
    if b[:2] == b"\x1f\x8b":                      # NBER stored some zips gzip-wrapped
        b = gzip.decompress(b)
    return zipfile.ZipFile(io.BytesIO(b))


def _csv(z: zipfile.ZipFile, name: str) -> pd.DataFrame:
    with z.open(name) as fh:
        d = pd.read_csv(fh, dtype=str, encoding="latin-1", keep_default_na=False,
                        na_values=[""])
    d.columns = [" ".join(c.split()) for c in d.columns]
    return d


def _num(s: pd.Series) -> pd.Series:
    """Numbers; '%', 'out of 10/30' stripped; 'No Score', '<11', 'Insufficient data' -> NaN."""
    s = s.astype(str).str.replace("%", "", regex=False)
    s = s.str.replace(r"\s*out of \d+", "", regex=True).str.strip()
    return pd.to_numeric(s, errors="coerce")


def _round_half_up(x) -> np.ndarray:
    return np.floor(np.asarray(x, float) + 0.5 + EPS)


# ------------------------------------------------------------------------------- rule

def reduction(tps, m) -> np.ndarray:
    """Payment reduction (%) from the TPS and the minimum TPS m, PY2014 on."""
    tps = np.asarray(tps, float)
    steps = np.ceil((np.asarray(m, float) - tps) / BAND - EPS)
    return np.where(tps >= m, 0.0, np.minimum(2.0, 0.5 * steps))


def reduction_2013(tps) -> np.ndarray:
    """PY2013 (0-30 scale): 30 -> 0; 26-29 -> 1.0; 21-25 -> 1.5; 0-20 -> 2.0 (76 FR 70261)."""
    t = np.asarray(tps, float)
    return np.select([t >= 30, t >= 26, t >= 21], [0.0, 1.0, 1.5], 2.0)


def measure_weights(S: dict[str, pd.DataFrame], spec: dict) -> dict[str, np.ndarray]:
    """Weights (share of the TPS) of each domain's measures after redistributing the weight
    of the measures and domains a facility has no score on. S[domain] holds that domain's
    measure scores (NaN = no score)."""
    W = {}
    doms = spec["domains"]
    if spec["redistribute"] == "prop":
        avail_w = {}
        for dom, (wd, ms) in doms.items():
            a = S[dom].notna().to_numpy()
            rel = a * np.array(list(ms.values()), float)
            tot = rel.sum(1, keepdims=True)
            W[dom] = rel / np.where(tot > 0, tot, 1)
            avail_w[dom] = np.where(tot[:, 0] > 0, wd, 0.0)
        total = sum(avail_w.values())
        for dom in doms:
            W[dom] = W[dom] * (avail_w[dom] / np.where(total > 0, total, 1))[:, None]
        return W
    left = np.ones(len(next(iter(S.values()))))
    n_avail = np.zeros_like(left)
    for dom, (wd, ms) in doms.items():
        a = S[dom].notna().to_numpy()
        rel = np.array(list(ms.values()), float)
        base = wd * rel / rel.sum()
        miss = ((~a) * base).sum(1, keepdims=True)
        na = a.sum(1, keepdims=True)
        W[dom] = a * (base + miss / np.where(na > 0, na, 1))
        left -= W[dom].sum(1)
        n_avail += a.sum(1)
    for dom in doms:
        a = S[dom].notna().to_numpy()
        W[dom] = a * (W[dom] + (left / np.where(n_avail > 0, n_avail, 1))[:, None])
    return W


def tps_from_measures(d: pd.DataFrame, py: int) -> pd.DataFrame:
    """The TPS recomputed from the published measure scores, and each domain's score on
    0-100 (the weighted mean of its measure scores with the within-domain weights)."""
    spec = SPEC[py]
    S = {dom: pd.DataFrame({c: _num(d[c]) for c in ms}) for dom, (w, ms) in spec["domains"].items()}
    W = measure_weights(S, spec)
    out = pd.DataFrame(index=d.index)
    total = np.zeros(len(d))
    for dom in spec["domains"]:
        s = S[dom].fillna(0).to_numpy()
        total += 10 * (np.round(W[dom], WEIGHT_DIGITS) * s).sum(1)
        w = W[dom].sum(1)
        out[f"dom_{dom}"] = np.where(w > 0, 10 * (W[dom] * s).sum(1) / np.where(w > 0, w, 1),
                                     np.nan)
        out[f"dom_{dom}_weight"] = w
    out["tps_recomputed"] = np.where(sum(S[dom].notna().any(axis=1) for dom in S) > 0,
                                     _round_half_up(total), np.nan)
    out["n_measures"] = sum(S[dom].notna().sum(1) for dom in S)
    return out


# ------------------------------------------------------------------------------ reading

def _read_qip(py: int) -> pd.DataFrame:
    z = _zip(py)
    name = next(n for n in z.namelist() if "QIP" in n)
    d = _csv(z, name)
    if py == 2013:
        out = pd.DataFrame({
            "ccn": d["CMS_Certification_Number"].str.strip().str.zfill(6),
            "network": _num(d["Network"]), "state": d["State"].str.strip(),
            "tps": _num(d["PY 2013_Total_Performance_Score"]),
            "reduction_published": _num(d["PY 2013_Payment_Reduction_Percentage"]),
            "cert_date": pd.to_datetime(d["DFC_Certification Date"], format="%m/%d/%Y",
                                        errors="coerce"),
            "chain_qip": d["Ownership on_12/31/2011 (from_SIMS database)"].fillna(""),
        })
        s = pd.DataFrame({"hgb": _num(d["HG>12 g/dL_Measure_Score"]),
                          "urr": _num(d["URR>=65%_Measure_Score"])})
        out["tps_recomputed"] = np.where(s.notna().all(1), _round_half_up(1.5 * s.sum(1)),
                                         np.nan)
        out["n_measures"] = s.notna().sum(1)
        out["py"] = py
        return out
    red = next(c for c in d.columns if "Payment Reduction Percentage" in c)
    own = next(c for c in d.columns if c.startswith("Ownership"))
    out = pd.DataFrame({
        "ccn": d["CMS Certification Number (CCN)"].str.strip().str.zfill(6),
        "network": _num(d["Network"]), "state": d["State"].str.strip(),
        "tps": _num(d["Total Performance Score"]),
        "reduction_published": _num(d[red].str.replace("No Reduction", "0", case=False)),
        "cert_date": pd.to_datetime(d["CMS Certification Date"], errors="coerce"),
        "chain_qip": d[own].fillna(""),
    })
    out = pd.concat([out, tps_from_measures(d, py)], axis=1)
    for k, v in _LAG[py].items():
        if isinstance(v, tuple):
            num, den = _num(d[v[0]]), _num(d[v[1]])
            out[f"lag_{k}"] = 100 * num / den.where(den > 0)
        else:
            out[f"lag_{k}"] = _num(d[v])
    out["py"] = py
    return out


def _read_dfc(py: int) -> pd.DataFrame:
    """Facility characteristics from the DFC release paired with the PY (download.py)."""
    z = _zip(py)
    name = next(n for n in z.namelist() if n.endswith("FAC_DATA.csv") and "CAHPS" not in n)
    d = _csv(z, name)
    col = {c.lower().rstrip("?"): c for c in d.columns}

    def get(*keys):
        for k in keys:
            if k in col:
                return d[col[k]]
        raise KeyError(keys)
    return pd.DataFrame({
        "ccn": get("provider number").str.strip().str.zfill(6),
        "for_profit": get("profit or non-profit").str.strip().eq("Profit").astype(float),
        "chain_owned": get("chain owned").str.strip().eq("Yes").astype(float),
        "chain_dfc": get("chain organization").fillna("").str.upper().str.strip(),
        "late_shift": _num(get("late shift")),
        "stations": _num(get("# of dialysis stations")),
        "offers_pd": _num(get("offers in-cener peritoneal dialysis",
                              "offers in-center peritoneal dialysis")),
        "offers_home_hd": _num(get("offers home hemodialysis training")),
    }).drop_duplicates("ccn")


# -------------------------------------------------------------------------------- build

def build() -> pd.DataFrame:
    """Every facility-PY row of the six QIP files, with the rule, the recomputed TPS and
    domain scores, the DFC characteristics, lags and the TPS of PY + 1 and PY + 2."""
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
    p = pd.read_csv(CACHE, dtype={"ccn": str, "state": str, "chain_qip": str,
                                  "chain_dfc": str}, keep_default_na=False, na_values=[""],
                    low_memory=False)
    for c in ["state", "chain_qip", "chain_dfc"]:
        p[c] = p[c].fillna("")
    p["cert_date"] = pd.to_datetime(p.cert_date)
    return p


def _build() -> pd.DataFrame:
    _require()
    frames = []
    for py in PYS:
        q = _read_qip(py)
        if py == 2013:              # only a lag and the rule check; its DFC file differs
            frames.append(q)
            continue
        f = _read_dfc(py)
        q["in_dfc"] = q.ccn.isin(f.ccn).astype(float)
        frames.append(q.merge(f, on="ccn", how="left"))
    p = pd.concat(frames, ignore_index=True, sort=False)
    p["min_tps"] = p.py.map(MIN_TPS).astype(float)
    p["q"] = p.tps - p.min_tps
    p["reduction_rule"] = np.where(p.py == 2013, reduction_2013(p.tps),
                                   reduction(p.tps, p.min_tps))
    p.loc[p.tps.isna(), "reduction_rule"] = np.nan
    p["band_rule"] = 4 - p.reduction_rule / 0.5
    p["band_published"] = 4 - p.reduction_published / 0.5
    # Values of other PYs, by CCN.
    key = p.set_index(["ccn", "py"])
    for lag, name in [(2, "tps_t2"), (1, "tps_t1")]:
        for col, out in [("tps", name), ("dom_clinical", f"clin_{name[-2:]}"),
                         ("reduction_rule", f"red_{name[-2:]}")]:
            idx = pd.MultiIndex.from_arrays([p.ccn, p.py + lag])
            p[out] = key[col].reindex(idx).to_numpy()
    idx = pd.MultiIndex.from_arrays([p.ccn, p.py - 2])
    p["tps_lag2"] = key["tps"].reindex(idx).to_numpy()
    p["present_t2"] = pd.MultiIndex.from_arrays([p.ccn, p.py + 2]).isin(key.index).astype(float)
    p["present_t1"] = pd.MultiIndex.from_arrays([p.ccn, p.py + 1]).isin(key.index).astype(float)
    return p.sort_values(["py", "ccn"]).reset_index(drop=True)


# ------------------------------------------------------------------------------- checks

def rule_check() -> dict:
    """The reduction rule on the published TPS against the published reduction, and the
    TPS recomputed from the measure scores against the published TPS, per PY."""
    p = build()
    out = {}
    for py in PYS:
        d = p[p.py == py]
        s = d[d.tps.notna()]
        r = {"facilities": int(len(d)), "scored": int(len(s)),
             "min_tps": MIN_TPS[py],
             "reduction_counts_published": s.reduction_published.value_counts().sort_index().to_dict(),
             "reduction_rule_agree": float((s.reduction_rule == s.reduction_published).mean()),
             "reduction_rule_mismatch": int((s.reduction_rule != s.reduction_published).sum()),
             "no_reduction_agree": float(((s.q >= 0) == (s.reduction_published == 0)).mean()),
             "tps_recomputed_agree": float((s.tps_recomputed == s.tps).mean()),
             "tps_recomputed_diff": (s.tps_recomputed - s.tps).value_counts().head(6).to_dict(),
             "unscored_published_reduction": d[d.tps.isna()].reduction_published
             .value_counts(dropna=False).to_dict(),
             "in_dfc": float(d.in_dfc.mean())}
        if py == 2013:
            m = s.reduction_rule != s.reduction_published
            r["mismatch_tps_values"] = s[m].tps.value_counts().to_dict()
        out[py] = r
    return out


# ------------------------------------------------------------------------------ samples

_CHAIN = {"DAVITA": "davita", "GAMBRO": "davita", "FRESENIUS": "fresenius",
          "RENAL CARE GROUP": "fresenius"}


def _chain(df: pd.DataFrame) -> pd.Series:
    name = df.chain_dfc.where(df.chain_dfc != "", df.chain_qip.str.upper())
    out = pd.Series("other_chain", index=df.index)
    for k, v in _CHAIN.items():
        out[name.str.contains(k, regex=False)] = v
    indep = (df.chain_owned == 0) | name.isin(["", "INDEPENDENT", "NOT A CHAIN"])
    out[indep & ~out.isin(["davita", "fresenius"])] = "independent"
    return out


def _frame(pys, y_col: str) -> pd.DataFrame:
    p = build()
    m = p.py.isin(pys) & p.tps.notna() & p[y_col].notna()
    return p[m].reset_index(drop=True)


def _design(df: pd.DataFrame) -> tuple:
    X = pd.DataFrame(index=df.index)
    pys = sorted(df.py.unique())
    for y in pys[1:]:
        X[f"py_{y}"] = (df.py == y).astype(float)
    X["for_profit"] = df.for_profit
    ch = _chain(df)
    for c in ["davita", "fresenius", "other_chain"]:
        X[f"chain_{c}"] = (ch == c).astype(float)
    X["log_stations"] = np.log(df.stations.clip(lower=1))
    X["late_shift"] = df.late_shift
    X["offers_pd"] = df.offers_pd
    X["offers_home_hd"] = df.offers_home_hd
    X["years_certified"] = (pd.to_datetime(df.py.astype(str) + "-01-01")
                            - df.cert_date).dt.days / 365.25
    X["not_in_dfc"] = 1 - df.in_dfc
    # Lagged rates and size: kept when observed in every PY of the sample.
    lag = {"lag_fistula": "lag_fistula_pct", "lag_catheter": "lag_catheter_pct",
           "lag_ktv_hd": "lag_ktv_hd_pct"}
    for c, name in lag.items():
        if c in df and df.groupby("py")[c].apply(lambda s: s.notna().mean()).min() > 0.5:
            X[name] = df[c]
            X[f"{name}_missing"] = df[c].isna().astype(float)
    if "lag_hd_pm" in df and df.groupby("py").lag_hd_pm.apply(lambda s: s.notna().mean()).min() > 0.5:
        X["log1p_lag_hd_patient_months"] = np.log1p(df.lag_hd_pm)
        X["log1p_lag_hd_patient_months_missing"] = df.lag_hd_pm.isna().astype(float)
        # PD patient-months: PY2015 reports 0 for facilities without PD; PY2017-2018 leave
        # them blank (no PD Kt/V score), read as 0. PY2014 has no PD column.
        if df.groupby("py").lag_pd_pm.apply(lambda s: (s > 0).mean()).min() > 0.05:
            pd_pm = df.lag_pd_pm.fillna(0)
            pm = df.lag_hd_pm.fillna(0) + pd_pm
            X["lag_pd_share"] = (pd_pm / pm.where(pm > 0)).to_numpy()
    if df.groupby("py").tps_lag2.apply(lambda s: s.notna().mean()).min() > 0.5:
        z = df.groupby("py").tps_lag2.transform(lambda s: (s - s.mean()) / s.std())
        X["tps_lag2_z"] = z
        X["tps_lag2_missing"] = z.isna().astype(float)
    for k in sorted(df.network.dropna().unique())[1:]:          # Network 1 base
        X[f"network_{int(k)}"] = (df.network == k).astype(float)
    # Missing values: column mean (the *_missing indicators mark the lagged ones).
    X = X.fillna(X.mean())
    X = X.loc[:, X.std() > 0]
    return X.to_numpy(float), list(X.columns)


def _extras(df: pd.DataFrame) -> dict:
    return {
        "ccn": df.ccn.to_numpy(str), "py": df.py.to_numpy(int), "year": df.py.to_numpy(int),
        "group": df.py.astype(str).to_numpy(),
        "group_cutoffs": {str(y): 0.0 for y in sorted(df.py.unique())},
        "state": df.state.to_numpy(str), "network": df.network.to_numpy(float),
        "tps": df.tps.to_numpy(float), "min_tps": df.min_tps.to_numpy(float),
        "tps_recomputed": df.tps_recomputed.to_numpy(float),
        "reduction_published": df.reduction_published.to_numpy(float),
        "reduction_rule": df.reduction_rule.to_numpy(float),
        "band_published": df.band_published.to_numpy(float),
        "dom_clinical": df.dom_clinical.to_numpy(float),
        "dom_reporting": df.dom_reporting.to_numpy(float),
        "y_tps_t2": df.tps_t2.to_numpy(float), "y_clinical_t2": df.clin_t2.to_numpy(float),
        "y_reduction_t2": df.red_t2.to_numpy(float),
        "y_tps_t1": df.tps_t1.to_numpy(float), "y_clinical_t1": df.clin_t1.to_numpy(float),
        "tps_lag2": df.tps_lag2.to_numpy(float),
        "chain": _chain(df).to_numpy(str),
        "rule": "reduction = 0 if TPS >= m, else min(2, 0.5 * ceil((m - TPS) / 10)) percent",
        "cutoffs": (0.0, -10.0, -20.0, -30.0),
    }


_CITATION = ("CMS ESRD QIP Complete QIP Data PY2013-PY2019 and Dialysis Facility Compare "
             "facility files (NBER mirror, data.nber.org/compare/dialysis); 42 CFR "
             "413.177-178; ESRD PPS final rules 76 FR 70228, 77 FR 67450, 79 FR 66120, "
             "80 FR 68968, 81 FR 77834")
_Y_LABEL = {"tps_t2": "TPS of PY + 2 (performance period = calendar year PY)",
            "clin_t2": "clinical domain score (0-100) of PY + 2",
            "tps_t1": "TPS of PY + 1 (performance period PY - 1, mostly pre-treatment)"}


def _single(pys, y_col: str, name: str) -> RDDSample:
    df = _frame(pys, y_col)
    X, names = _design(df)
    ex = _extras(df)
    ex.update({"cutoff": 0.0,
               "eligible_published": (df.reduction_published == 0).to_numpy(float),
               "treated_side": "no payment reduction: TPS >= minimum TPS (Q >= 0)"})
    return RDDSample(
        Q=df.q.to_numpy(float), X=X, Y=df[y_col].to_numpy(float), threshold=0.0,
        name=name, feature_names=names,
        description=(f"Medicare ESRD QIP dialysis facilities, PY {', '.join(map(str, pys))}. "
                     "Q = TPS - minimum TPS of the PY; treatment = 1{Q >= 0}, no payment "
                     f"reduction (0.5-2% below); Y = {_Y_LABEL[y_col]}."),
        citation=_CITATION, treatment_rule=lambda Q: (np.asarray(Q) >= 0).astype(int),
        extras=ex,
    )


def load() -> RDDSample:
    """Q = TPS - minimum TPS, PY2015 and PY2017; D = 1{Q >= 0}, no payment reduction;
    Y = TPS of PY + 2."""
    return _single(MAIN_PYS, "tps_t2", "esrd_qip")


def load_clinical() -> RDDSample:
    """As ``load()``; Y = clinical domain score of PY + 2 (0-100)."""
    return _single(MAIN_PYS, "clin_t2", "esrd_qip_clinical")


def load_next_py() -> RDDSample:
    """As ``load()`` for PY2014, PY2017, PY2018; Y = TPS of PY + 1. Its performance period
    (PY - 1) ends before the final PY scores are published: a placebo-like outcome."""
    return _single(NEXT_PYS, "tps_t1", "esrd_qip_next_py")


def load_bands() -> RDDSample:
    """All four cut points on Q = TPS - m: D = #{c in (0, -10, -20, -30): Q >= c}, 0-4
    (4 = no reduction, 0 = 2%); reduction = 0.5 * (4 - D) %. Same sample, X, Y as
    ``load()``."""
    df = _frame(MAIN_PYS, "tps_t2")
    X, names = _design(df)
    cuts = np.array([0.0, -10.0, -20.0, -30.0])
    ex = _extras(df)
    ex.update({"cutoff": 0.0,
               "eligible_published": df.band_published.to_numpy(float),
               "treated_side": "D = number of cut points at or below Q; higher D = smaller "
                               "reduction"})
    return RDDSample(
        Q=df.q.to_numpy(float), X=X, Y=df.tps_t2.to_numpy(float),
        threshold=tuple(cuts), name="esrd_qip_bands", feature_names=names,
        description=("Medicare ESRD QIP dialysis facilities, PY2015 and PY2017. Q = TPS - "
                     "minimum TPS; D = number of the cut points 0, -10, -20, -30 at or "
                     "below Q (payment reduction 0.5 * (4 - D) %); Y = TPS of PY + 2."),
        citation=_CITATION,
        treatment_rule=lambda Q: (np.asarray(Q, float)[:, None] >= cuts[None, :]).sum(1),
        extras=ex,
    )


def load_two_score() -> RDDSample:
    """Q = (clinical, reporting) domain scores, PY2015 and PY2017 (weights 0.75 / 0.25 in
    both); D = 1{0.75 Q1 + 0.25 Q2 >= 59.5}, no payment reduction; Y = TPS of PY + 2."""
    df = _frame(MAIN_PYS, "tps_t2")
    df = df[df.dom_clinical.notna() & df.dom_reporting.notna()].reset_index(drop=True)
    X, names = _design(df)
    w = np.array(TWO_SCORE_WEIGHTS)
    cut = MIN_TPS[2015] - 0.5
    Q = df[["dom_clinical", "dom_reporting"]].to_numpy(float)
    ex = _extras(df)
    ex.update({"weights": TWO_SCORE_WEIGHTS, "cutoff": cut, "offset": np.zeros(len(df)),
               "eligible_published": (df.reduction_published == 0).to_numpy(float),
               "treated_side": f"no payment reduction: 0.75 Q1 + 0.25 Q2 >= {cut} (the TPS, "
                               "rounded half up, at least 60)"})
    return RDDSample(
        Q=Q, X=X, Y=df.tps_t2.to_numpy(float), threshold=(cut, cut),
        name="esrd_qip_two_score", feature_names=names,
        description=("Medicare ESRD QIP dialysis facilities, PY2015 and PY2017. Q = "
                     "(clinical domain score, reporting domain score), 0-100; treatment = "
                     f"1{{0.75 Q1 + 0.25 Q2 >= {cut}}}, no payment reduction; Y = TPS of "
                     "PY + 2."),
        citation=_CITATION,
        treatment_rule=lambda Q: (np.asarray(Q, float) @ w >= cut - EPS).astype(int),
        extras=ex,
    )
