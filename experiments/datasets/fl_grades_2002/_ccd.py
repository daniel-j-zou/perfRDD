"""Florida rows of the NCES Common Core of Data public school universe, 2001-02 to 2013-14.

2001-02 to 2006-07 are fixed-width files: the column positions are read from the
record layout NCES publishes next to each file (lines ``NAME START END LENGTH TYPE``).
2007-08 onward are tab-delimited with a header row; the year suffix of the 2007-08 to
2009-10 variable names (``MEMBER07``) is dropped. The zip members are read in place.

Returned per school: the state's district and school numbers (``STID`` and the last
four digits of ``SEASCH``, which match FLDOE's district and school numbers), membership,
free and reduced-price lunch counts, race counts, FTE teachers, Title I, magnet and
charter flags, school level and NCES locale. Negative or non-numeric counts (CCD's
missing and not-applicable codes) become NaN.
"""
from __future__ import annotations

import io
import re
import zipfile

import numpy as np
import pandas as pd

from experiments.datasets.fl_grades_2002.download import ccd_paths

FLORIDA = "12"
_KEEP = ["NCESSCH", "STID", "SEASCH", "TYPE", "STATUS", "LEVEL", "TITLEI", "MAGNET",
         "CHARTR", "FTE", "FRELCH", "REDLCH", "TOTFRL", "MEMBER", "AM", "ASIAN", "HISP",
         "BLACK", "WHITE", "PACIFIC", "TR", "LOCALE", "ULOCAL"]
_NUM = ["FTE", "FRELCH", "REDLCH", "TOTFRL", "MEMBER", "AM", "ASIAN", "HISP", "BLACK",
        "WHITE", "PACIFIC", "TR"]


def _layout(path) -> dict[str, tuple[int, int, bool]]:
    """name (year suffix dropped) -> (start, end, decimal flag) from a record layout."""
    out = {}
    pat = re.compile(r"^\s*([A-Z][A-Z0-9]*?)(\d{2})?\s+(\d{4})\s+(\d{4})\s+(\d+)(\*?)\s")
    for line in path.read_bytes().decode("latin-1").splitlines():
        m = pat.match(line)
        if not m:
            continue
        name = m.group(1) if m.group(1) != "NCESSCH" else "NCESSCH"
        if line.lstrip().startswith("NCESSCH"):
            name = "NCESSCH"
        if name in _KEEP and name not in out:
            out[name] = (int(m.group(3)), int(m.group(4)), bool(m.group(6)))
    return out


def _member(z: zipfile.ZipFile) -> str:
    return next(n for n in z.namelist() if n.lower().endswith((".dat", ".txt")))


def _fixed(year: int) -> pd.DataFrame:
    data, layout = ccd_paths(year)
    lay = _layout(layout)
    rows = []
    with zipfile.ZipFile(data) as z, z.open(_member(z)) as fh:
        for raw in io.TextIOWrapper(fh, encoding="latin-1"):
            if raw[:2] != FLORIDA:
                continue
            rows.append({k: raw[s - 1:e].strip() for k, (s, e, _) in lay.items()})
    # Fields marked "*" in the layout carry an explicit decimal point, so no rescaling.
    return pd.DataFrame(rows)


def _tabbed(year: int) -> pd.DataFrame:
    data, _ = ccd_paths(year)
    with zipfile.ZipFile(data) as z, z.open(_member(z)) as fh:
        head = fh.readline().decode("latin-1").rstrip("\r\n").split("\t")
    names = {}
    for h in head:
        base = re.sub(r"\d{2}$", "", h) if h not in ("NCESSCH",) else h
        if base in _KEEP and base not in names.values():
            names[h] = base
    out = []
    with zipfile.ZipFile(data) as z, z.open(_member(z)) as fh:
        for ch in pd.read_csv(fh, sep="\t", dtype=str, usecols=list(names),
                              encoding="latin-1", chunksize=50_000):
            ch = ch.rename(columns=names)
            out.append(ch[ch.NCESSCH.str[:2] == FLORIDA])
    return pd.concat(out, ignore_index=True)


def read_ccd(year: int) -> pd.DataFrame:
    """Florida schools in the CCD file for the school year ending in spring `year`."""
    df = _fixed(year) if year <= 2007 else _tabbed(year)
    for c in _NUM:
        v = pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
        df[c] = v.where(v >= 0) if isinstance(v, pd.Series) else np.nan
    seasch = df.SEASCH.astype(str).str.strip()
    out = pd.DataFrame({
        "year": year,
        "district": pd.to_numeric(df.STID.astype(str).str.strip(), errors="coerce"),
        "school": pd.to_numeric(seasch.str[-4:], errors="coerce"),
        "ncessch": df.NCESSCH.astype(str).str.strip(),
        "ccd_type": df.TYPE.astype(str).str.strip(),
        "ccd_level": df.LEVEL.astype(str).str.strip(),
        "ccd_title1": df.TITLEI.astype(str).str.strip(),
        "ccd_magnet": df.MAGNET.astype(str).str.strip(),
        "ccd_charter": df.CHARTR.astype(str).str.strip(),
        "ccd_locale": (df.ULOCAL if "ULOCAL" in df else df.LOCALE).astype(str).str.strip(),
        "member": df.MEMBER, "frl_count": df.TOTFRL.fillna(df.FRELCH + df.REDLCH),
        "fte": df.FTE,
    })
    race = df[[c for c in ["AM", "ASIAN", "HISP", "BLACK", "WHITE", "PACIFIC", "TR"] if c in df]]
    out["white"] = df.WHITE
    out["black"] = df.BLACK
    out["hisp"] = df.HISP
    out["race_total"] = race.sum(axis=1, min_count=1)
    out = out.dropna(subset=["district", "school"])
    out = out[~out.duplicated(["district", "school"], keep=False)]
    return out.astype({"district": int, "school": int}).reset_index(drop=True)
