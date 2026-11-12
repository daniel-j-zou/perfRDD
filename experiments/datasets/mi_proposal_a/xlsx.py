"""A minimal, dependency-free reader for the cell values of an .xlsx workbook.

The research environment has no openpyxl / xlrd, and the CDE workbooks are plain
Office Open XML: a zip of XML parts. This reads the cached cell values (not formulas),
shared strings and inline strings of one sheet into a list of rows. It never evaluates
anything; the XML is parsed with the standard-library expat parser (2.6, which limits
entity expansion).
"""
from __future__ import annotations

import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
      "pr": "http://schemas.openxmlformats.org/package/2006/relationships"}
_M = "{%s}" % NS["m"]
_COL = re.compile(r"([A-Z]+)(\d+)")


def _col_index(ref: str) -> int:
    letters = _COL.match(ref).group(1)
    k = 0
    for ch in letters:
        k = 26 * k + (ord(ch) - 64)
    return k - 1


def sheet_names(path: Path) -> List[str]:
    with zipfile.ZipFile(path) as z:
        wb = ET.fromstring(z.read("xl/workbook.xml"))
    return [s.get("name") for s in wb.find("m:sheets", NS)]


def _sheet_part(z: zipfile.ZipFile, name: str) -> str:
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    rid = None
    for s in wb.find("m:sheets", NS):
        if s.get("name") == name:
            rid = s.get("{%s}id" % NS["r"])
    if rid is None:
        raise KeyError(f"no sheet {name!r}")
    for r in rels:
        if r.get("Id") == rid:
            target = r.get("Target").lstrip("/")
            return target if target.startswith("xl/") else "xl/" + target
    raise KeyError(rid)


def _shared_strings(z: zipfile.ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in z.namelist():
        return []
    out = []
    for si in ET.fromstring(z.read("xl/sharedStrings.xml")).iter(_M + "si"):
        out.append("".join(t.text or "" for t in si.iter(_M + "t")))
    return out


def read_sheet(path: Path, name: Optional[str] = None, index: int = 0) -> List[list]:
    """Rows of one sheet (by name, else by position) as lists of str / float / None."""
    path = Path(path)
    with zipfile.ZipFile(path) as z:
        if name is None:
            name = sheet_names(path)[index]
        part = _sheet_part(z, name)
        sst = _shared_strings(z)
        rows: List[list] = []
        with z.open(part) as fh:
            for _, el in ET.iterparse(fh, events=("end",)):
                if el.tag != _M + "row":
                    continue
                r = int(el.get("r")) - 1
                vals: dict = {}
                for c in el.findall(_M + "c"):
                    t = c.get("t")
                    v = c.find(_M + "v")
                    if t == "s":
                        val = sst[int(v.text)] if v is not None else None
                    elif t == "inlineStr":
                        val = "".join(x.text or "" for x in c.iter(_M + "t"))
                    elif t in ("str", "e"):
                        val = v.text if v is not None else None
                    elif t == "b":
                        val = float(v.text) if v is not None else None
                    else:
                        val = float(v.text) if v is not None and v.text not in (None, "") else None
                    vals[_col_index(c.get("r"))] = val
                el.clear()
                while len(rows) < r:
                    rows.append([])
                width = max(vals) + 1 if vals else 0
                rows.append([vals.get(j) for j in range(width)])
    return rows
