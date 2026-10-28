"""Read the ESRD QIP Performance Score Summary Report (PSSR) files into one format.

``read(py)`` returns one row per facility for payment year ``py`` (2012-2026) with:

    ccn, py, name, state, zip, network, tps, red_pub (published payment reduction, in
    percent; NaN when the file says N/A), red_label (the raw text), nat_avg_tps,
    cert_date, owner (corporate ownership at the end of the performance year before),
    ich_ach / ich_imp (in-center HD offered, PY 2019+), and measure scores ``m_<name>``.

The files differ by year (PY 2012-2013 and 2016-2026 are workbooks, PY 2014, 2015 and
2017 are CSVs), so each year has its own column map below.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from experiments.datasets.cms_esrd_qip.xlsx import read_xlsx, sheets

RAW = Path(__file__).parent / "data" / "raw" / "pssr"

FILES = {
    2012: "pssr_py2012/ESRD QIP PY 2012 Performance.xlsx",
    2013: "pssr_py2013/ESRD QIP PY 2013 Performance.xlsx",
    2014: "pssr_py2014/QIP-PY14-Performance/Complete Dataset.csv",
    2015: "pssr_py2015/PY2015_PSSR_Complete Dataset.csv",
    2016: "pssr_py2016/PY 2016 Performance Score Summary Report - Final.xlsx",
    2017: "pssr_py2017/ESRD QIP-Complete_Data_20170111.csv",
    2018: "pssr_py2018/ESRD QIP PY 2018 Performance Score Summary Report - Final.XLSX",
    2019: "pssr_py2019/Performance Score Summary Report - Final.XLSX",
    2020: "pssr_py2020/ESRD_QIP_PY_2020_PSSR_File.xlsx",
    2021: "pssr_py2021/ALL-CCNS-FINAL-PSSR-PY2021_vFINAL(508).xlsx",
    2022: "pssr_py2022/ALL-CCNS-FINAL-PSSR-PY2022(508).xlsx",
    2023: "pssr_py2023/ALL-CCNS-FINAL-PSSR-PY2023_508.xlsx",
    2024: "pssr_py2024/ALL-CCNS-FINAL-PSSR-PY2024_vFINAL(508).xlsx",
    2025: "pssr_py2025/ALL-CCNS-FINAL-PSSR-PY2025.xlsx",
    2026: "pssr_py2026/ALL-CCNS-FINAL-PSSR-PY2026.xlsx",
}

# Standard measure names -> regular expressions matched against (normalised) headers.
# Only "Measure Score" / "Topic Score" columns are used, not rates or "applied" labels.
MEASURES = {
    "hgb12": r"^(hg|hgb|hemoglobin) ?>? ?12 ?g/?dl (measure|component|performance)? ?score$",
    "hgb10": r"^hg ?< ?10 g/dl component score$",
    "urr": r"^urr( ?>= ?65%)? (measure|component)? ?score$",
    "ktv": r"^kt/v ((measure topic|comprehensive|dialysis adequacy combined) ?(measure )?score)$",
    "ktv_hd": r"^kt/v -* ?adult (hd|hemodialysis) measure score$",
    "ktv_pd": r"^kt/v -* ?adult (pd|peritoneal dialysis) measure score$",
    "vat": r"^(vat measure topic score|vat combined measure score|vascular access combined measure score)$",
    "fistula": r"^((vat -* ?)?fistula measure score|standardized fistula rate measure score)$",
    "catheter": r"^((vat -* ?)?catheter (rate )?measure score)$",
    "hypercal": r"^hypercalcemia measure score$",
    "nhsn": r"^(nhsn (bsi )?measure score|nhsn measure topic score)$",
    "nhsn_rep": r"^(nhsn (dialysis )?event reporting score|nhsn dialysis event measure score)$",
    "srr": r"^(standardized readmission ratio \(srr\) )?srr measure score$",
    "strr": r"^strr measure score$",
    "shr": r"^shr measure score$",
    "ichcahps": r"^ich ?cahps (admin |reporting )?(measure )?score$",
    "mineral": r"^mineral metabolism (reporting )?(measure )?score$",
    "anemia": r"^anemia management (reporting )?(measure )?score$",
    "depression": r"^clinical depression (measure )?score$",
    "pain": r"^pain assessment measure score$",
    "flu": r"^nhsn influenza measure score$",
    "phos": r"^serum phosphorus measure score$",
    "ufr": r"^ufr measure score$",
    "pppw": r"^pppw measure score$",
    "medrec": r"^medication reconciliation measure score$",
    "covid_hcp": r"^covid-19 hcp vaccination measure score$",
    "fche": r"^facility commitment to health equity measure score$",
}


def _norm(h) -> str:
    h = "" if h is None else str(h)
    return re.sub(r"\s+", " ", h.replace("\n", " ")).strip().lower()


def _table(py: int) -> pd.DataFrame:
    f = RAW / FILES[py]
    if not f.exists():
        raise FileNotFoundError(f"{f} missing; run the download script (see README).")
    if f.suffix.lower() == ".csv":
        return pd.read_csv(f, dtype=str, encoding="latin-1", keep_default_na=False)
    rows = read_xlsx(f, next(iter(sheets(f))))
    # header row: the first row with 'Facility Name' in column 0
    k = next(i for i, r in enumerate(rows) if r and _norm(r[0]) == "facility name")
    hdr = [_norm(h) for h in rows[k]]
    body = [r + [None] * (len(hdr) - len(r)) for r in rows[k + 1:]]
    df = pd.DataFrame([r[:len(hdr)] for r in body], columns=hdr)
    return df.fillna("")


def _col(df: pd.DataFrame, *pats: str) -> pd.Series:
    cols = {_norm(c): c for c in df.columns}
    for p in pats:
        hit = [c for n, c in cols.items() if re.search(p, n)]
        if hit:
            return df[hit[0]]
    return pd.Series([""] * len(df), index=df.index)


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(), errors="coerce")


def _reduction(s: pd.Series, py: int) -> pd.Series:
    t = s.astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=s.index)
    out[t.isin(["no reduction", "0", "0.0%", "0%", "0.00%"])] = 0.0
    pct = t.str.extract(r"^([0-9.]+)\s*%$")[0]
    out = out.where(pct.isna(), pd.to_numeric(pct, errors="coerce"))
    frac = pd.to_numeric(t.where(~t.str.contains("%")), errors="coerce")
    # Workbooks store reductions as fractions (0.005); CSVs as text "0.5%".
    out = out.where(frac.isna() | out.notna(), (100 * frac).round(2))
    return out


def read(py: int) -> pd.DataFrame:
    df = _table(py)
    out = pd.DataFrame({
        "ccn": _col(df, r"^(ccn|cms certification number( \(ccn\))?)$").astype(str)
        .str.strip().str.zfill(6),
        "name": _col(df, r"^facility name$"),
        "state": _col(df, r"^(physical address )?state$"),
        "zip": _col(df, r"^(physical address )?zip code$").astype(str).str[:5],
        "network": _num(_col(df, r"^network( number)?$")),
        "tps": _num(_col(df, r"^(py \d{4} )?total performance score$")),
        "red_label": _col(df, r"payment reduction percentage$"),
        "nat_avg_tps": _num(_col(df, r"^national av(g|erage) total performance score$")),
        "cert_date": pd.to_datetime(
            _col(df, r"^(dfc |cms )?certification date$"), errors="coerce"),
        "owner": _col(df, r"^(corporate )?ownership (as of|on)").astype(str).str.strip(),
        "ich_ach": _col(df, r"^in.center hemodialysis offered .*achievement period"),
        "ich_imp": _col(df, r"^in.center hemodialysis offered .*improvement period"),
    })
    out.insert(1, "py", py)
    out["red_pub"] = _reduction(out["red_label"], py)
    # Workbook dates may be Excel serial numbers.
    raw_cd = _col(df, r"^(dfc |cms )?certification date$")
    serial = _num(raw_cd)
    if serial.notna().mean() > 0.5:
        out["cert_date"] = pd.to_datetime("1899-12-30") + pd.to_timedelta(serial, unit="D")
    for name, pat in MEASURES.items():
        s = _col(df, pat)
        out[f"m_{name}"] = _num(s)
    out = out[out["ccn"].str.fullmatch(r"\d{6}") & (out["ccn"] != "000000")
              & (out["name"].astype(str) != "")]
    return out.reset_index(drop=True)


def _block(rows: List[list], title: str) -> pd.DataFrame:
    """One measure block of the PY 2016 'Clinical Measure Detail' sheet."""
    start = next(i for i, r in enumerate(rows) if r and str(r[0]).startswith("Measure Name")
                 and title in str(r[0]))
    k = next(i for i in range(start, len(rows)) if rows[i] and _norm(rows[i][0]) == "facility name")
    hdr = [_norm(h) for h in rows[k]]
    body = []
    for r in rows[k + 1:]:
        if r and r[0] and str(r[0]).startswith("Measure Name"):
            break
        body.append((r + [None] * len(hdr))[:len(hdr)])
    return pd.DataFrame(body, columns=hdr).fillna("")


def patients(py: int) -> pd.DataFrame:
    """Eligible dialysis patients in the score's performance (achievement) period.

    PY 2012-2014: URR patients (in-center HD); PY 2015-2018: Kt/V adult HD patients;
    PY 2019+: Kt/V comprehensive (all modalities and ages). '<11' and blanks -> NaN.
    """
    f = RAW / FILES[py]
    if py in (2012, 2013, 2014, 2015, 2017):
        df = _table(py)
        pat = {2012: r"number of eligible urr patients, 2010",
               2013: r"number of eligible urr patients, 2011",
               2014: r"number of urr eligible patients performance period",
               2015: r"number of patients included in kt/v adult hemodialysis measure score performance period",
               2017: r"number of patients included in kt/v adult hemodialysis measure score achievement period"}[py]
    else:
        names = list(sheets(f))
        if py == 2016:
            df = _block(read_xlsx(f, "Clinical Measure Detail"), "Kt/V Dialysis Adequacy: Hemodialysis")
        else:
            sh = next(s for s in names if s.upper() in ("KTV HEMODIALYSIS", "KTV COMPREHENSIVE"))
            rows = read_xlsx(f, sh)
            k = next(i for i, r in enumerate(rows) if r and _norm(r[0]) == "facility name")
            hdr = [_norm(h) for h in rows[k]]
            df = pd.DataFrame([(r + [None] * len(hdr))[:len(hdr)] for r in rows[k + 1:]],
                              columns=hdr).fillna("")
        pat = r"^number of eligible patients \(?achievement\)?$"
    out = pd.DataFrame({
        "ccn": _col(df, r"^(ccn|cms certification number( \(ccn\))?)$").astype(str)
        .str.strip().str.zfill(6),
        "n_patients": _num(_col(df, pat)),
    })
    out = out[out["ccn"].str.fullmatch(r"\d{6}") & (out["ccn"] != "000000")]
    out = out.drop_duplicates("ccn")
    out.insert(1, "py", py)
    return out.reset_index(drop=True)


def read_all(years=tuple(FILES)) -> pd.DataFrame:
    return pd.concat([read(py) for py in years], ignore_index=True)
