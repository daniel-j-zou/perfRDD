"""Stream selected columns of one worksheet of a large .xlsx file.

The Report Card Public Data Set has one row per school, district and the state, and
900-1,300 columns on its "General" sheet (up to 75 MB of sheet XML). The project reader
``fl_school_grades/_spreadsheet.py`` builds the whole grid in memory, so this module
reads the sheet with ``iterparse`` and keeps only the cells of the requested columns.
Standard library only; the environment has no openpyxl.
"""
from __future__ import annotations

import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

_M = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def _col(ref: str) -> int:
    n = 0
    for ch in ref:
        if not ch.isalpha():
            break
        n = 26 * n + ord(ch.upper()) - 64
    return n - 1


def _sheet_part(z: zipfile.ZipFile, name: str) -> str:
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    target = {r.get("Id"): r.get("Target") for r in rels}
    for sh in wb.find(f"{_M}sheets"):
        if sh.get("name") == name:
            t = target[sh.get(f"{_R}id")].lstrip("/")
            return t if t.startswith("xl/") else f"xl/{t}"
    raise KeyError(f"no sheet {name!r}")


def _shared_strings(z: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in z.namelist():
        return []
    out = []
    with z.open("xl/sharedStrings.xml") as fh:
        for _, el in ET.iterparse(fh):
            if el.tag == f"{_M}si":
                out.append("".join(x.text or "" for x in el.iter(f"{_M}t")
                                   if x not in el.findall(f".//{_M}rPh/{_M}t")))
                el.clear()
    return out


def _value(c, sst):
    kind, v = c.get("t"), c.find(f"{_M}v")
    if kind == "inlineStr":
        return "".join(x.text or "" for x in c.iter(f"{_M}t"))
    if v is None or v.text is None:
        return None
    if kind == "s":
        return sst[int(v.text)]
    if kind in ("str", "e"):
        return v.text
    return float(v.text)


def read_columns(path: Path, sheet: str, columns: dict[str, list[str]]) -> list[dict]:
    """Rows of ``sheet`` as dicts {key: value} for each key in ``columns``.

    ``columns`` maps an output key to the header names it may have (the first header
    row is row 1; names are compared after collapsing whitespace). A key whose header
    is absent is returned as None in every row. Header matching is exact, so aliases
    must list every spelling used across years.
    """
    with zipfile.ZipFile(path) as z:
        sst = _shared_strings(z)
        part = _sheet_part(z, sheet)
        want: dict[int, str] = {}
        rows: list[dict] = []
        with z.open(part) as fh:
            header = None
            for _, el in ET.iterparse(fh):
                if el.tag != f"{_M}row":
                    continue
                cells = el.findall(f"{_M}c")
                if header is None:
                    header = {_col(c.get("r")): " ".join(str(_value(c, sst) or "").split())
                              for c in cells}
                    names = {v: k for k, v in header.items()}
                    for key, aliases in columns.items():
                        for a in aliases:
                            if a in names:
                                want[names[a]] = key
                                break
                else:
                    rec = dict.fromkeys(columns)
                    for c in cells:
                        j = _col(c.get("r"))
                        if j in want:
                            rec[want[j]] = _value(c, sst)
                    rows.append(rec)
                el.clear()
    return rows
