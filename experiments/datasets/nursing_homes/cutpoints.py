"""CMS's monthly state cut points for the health-inspection star rating.

Two public sources, both downloaded by ``download.py``:

* ``Health-Inspection-Cut-Points-2008-2016.zip`` (www.cms.gov): one SAS file per year with
  the state percentiles of the weighted health-inspection score, January 2008 - October
  2016 (``defscore_SP10``, ``_SP33``, ``_SP57``, ``_SP80``; ``_NP*`` national).
* ``cutpointstable.pdf``, CMS's monthly "State-Level Health Inspection Cut Point Table",
  September 2016 - November 2019, as captured by the Internet Archive. The adapter does
  not parse PDFs; ``build_pdf_table()`` turns them into ``state_cutpoints_2016_2019.csv``
  (in this folder, tracked) with poppler's ``pdftotext -layout``. Regenerate with

      python -m experiments.datasets.nursing_homes.cutpoints

  The two sources overlap in September and October 2016 and agree there (see README).

Each month's table lists, per state, the upper score of 5, 4, 3 and 2 stars. The tables'
notes give the rule: 5 stars <= P10, 4 stars in (P10, P33.33], 3 stars in (P33.33, P56.67],
2 stars in (P56.67, P80], 1 star > P80, where P is the state's percentile of the weighted
health-inspection score (lower is better). Guam, the Virgin Islands and (in some months)
Puerto Rico use the national distribution.
"""
from __future__ import annotations

import io
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw" / "cutpoints"
SAS_ZIP = RAW / "Health-Inspection-Cut-Points-2008-2016.zip"
PDF_TABLE = HERE / "state_cutpoints_2016_2019.csv"
COLS = ["c5", "c4", "c3", "c2"]          # upper score of 5, 4, 3, 2 stars (P10, P33, P57, P80)

STATES = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "D. C.": "DC", "Delaware": "DE", "Florida": "FL",
    "Georgia": "GA", "Guam": "GU", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL",
    "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI",
    "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
    "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
    "New Mexico": "NM", "New York": "NY", "North Carolina": "NC", "North Dakota": "ND",
    "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
    "Puerto Rico": "PR", "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
    "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
    "Virgin Islands": "VI", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY",
}
_ROW = re.compile(r"^\s*(?P<state>[A-Z][A-Za-z. ]+?)\s+(?P<n>[\d,]+)\*?\s+"
                  r">(?P<c2>[\d.]+)\s+<[\d.]+\s+>(?P<c3>[\d.]+)\s+<[\d.]+\s+"
                  r">(?P<c4>[\d.]+)\s+<[\d.]+\s+>(?P<c5>[\d.]+)\s+<[\d.]+\s*$")
_MONTH = re.compile(r"by State - \((\d\d)-01-(\d{4})\)")


def sas_table() -> pd.DataFrame:
    """Jan 2008 - Oct 2016, from CMS's SAS files."""
    z = zipfile.ZipFile(SAS_ZIP)
    out = []
    for name in sorted(n for n in z.namelist() if n.endswith(".sas7bdat")):
        d = pd.read_sas(io.BytesIO(z.read(name)), format="sas7bdat", encoding="latin-1")
        out.append(pd.DataFrame({
            "month": [f"{int(y)}-{int(m):02d}" for y, m in zip(d.year, d.month)],
            "state": d.STATE.str.strip(), "n_fac": d.numfac_s,
            "c5": d.defscore_SP10, "c4": d.defscore_SP33, "c3": d.defscore_SP57,
            "c2": d.defscore_SP80, "source": name}))
    return pd.concat(out, ignore_index=True)


def parse_pdf_text(text: str, source: str) -> pd.DataFrame:
    m = _MONTH.search(text)
    if not m:
        raise ValueError(f"{source}: no table month")
    month = f"{m.group(2)}-{m.group(1)}"
    rows, seen = [], set()
    for line in text.splitlines():
        r = _ROW.match(line)
        if not r or r.group("state").strip() not in STATES:
            continue
        st = STATES[r.group("state").strip()]
        if st in seen:                    # a stray repeated row (February 2017)
            continue
        seen.add(st)
        rows.append({"month": month, "state": st, "n_fac": int(r.group("n").replace(",", "")),
                     **{c: float(r.group(c)) for c in COLS}, "source": source})
    return pd.DataFrame(rows)


def build_pdf_table() -> pd.DataFrame:
    """Parse every downloaded ``cutpointstable_*.pdf`` with ``pdftotext -layout`` and write
    ``state_cutpoints_2016_2019.csv``."""
    exe = shutil.which("pdftotext")
    if exe is None:
        raise RuntimeError("pdftotext (poppler) is needed to rebuild the PDF table")
    out = []
    for pdf in sorted(RAW.glob("cutpointstable_*.pdf")):
        text = subprocess.run([exe, "-layout", str(pdf), "-"], check=True,
                              capture_output=True).stdout.decode("utf-8", "replace")
        out.append(parse_pdf_text(text, pdf.name))
    df = pd.concat(out, ignore_index=True).sort_values(["month", "state"])
    if df.duplicated(["month", "state"]).any():
        raise ValueError("two PDFs for the same month")
    df.to_csv(PDF_TABLE, index=False, float_format="%.3f")
    return df


def table() -> pd.DataFrame:
    """All months: the SAS files through October 2016, the PDF table from November 2016.
    Columns: month (YYYY-MM), state, n_fac, c5, c4, c3, c2 (upper score of each star
    level; 1 star is score > c2), source."""
    sas = sas_table()
    pdf = pd.read_csv(PDF_TABLE)
    pdf = pdf[pdf.month > sas.month.max()]
    return pd.concat([sas, pdf], ignore_index=True).sort_values(
        ["month", "state"]).reset_index(drop=True)


if __name__ == "__main__":
    t = build_pdf_table()
    print(t.groupby("month").state.size().to_string())
    s = sas_table()
    both = s.merge(t, on=["month", "state"], suffixes=("_sas", "_pdf"))
    for c in COLS:
        d = (both[f"{c}_sas"] - both[f"{c}_pdf"]).abs()
        print(c, "overlap rows", len(both), "max |sas - pdf|", round(float(d.max()), 4))
