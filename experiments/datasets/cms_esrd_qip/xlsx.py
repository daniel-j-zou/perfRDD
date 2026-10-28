"""Minimal .xlsx reader (standard library only: zipfile + ElementTree).

The PSSR workbooks are plain Office Open XML spreadsheets. This reads cell values as
strings (shared strings resolved, numbers as written in the XML), which is all the
adapter needs, without adding openpyxl to the environment.
"""
from __future__ import annotations

import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
RNS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
PNS = "{http://schemas.openxmlformats.org/package/2006/relationships}"


def _col(ref: str) -> int:
    n = 0
    for ch in re.match(r"[A-Z]+", ref).group(0):
        n = n * 26 + ord(ch) - 64
    return n - 1


def _shared(zf: zipfile.ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    out = []
    with zf.open("xl/sharedStrings.xml") as f:
        for _, el in ET.iterparse(f):
            if el.tag == NS + "si":
                out.append("".join(t.text or "" for t in el.iter(NS + "t")))
                el.clear()
    return out


def sheets(path: Path) -> Dict[str, str]:
    """Sheet name -> member path inside the ZIP."""
    with zipfile.ZipFile(path) as zf:
        wb = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    tgt = {r.get("Id"): r.get("Target") for r in rels.iter(PNS + "Relationship")}
    out = {}
    for s in wb.iter(NS + "sheet"):
        t = tgt[s.get(RNS + "id")].lstrip("/")
        out[s.get("name")] = t if t.startswith("xl/") else "xl/" + t
    return out


def read_xlsx(path: Path, sheet: Optional[str] = None, max_rows: Optional[int] = None
              ) -> List[List[Optional[str]]]:
    """Rows of one sheet (default: the first) as lists of strings (None when empty)."""
    path = Path(path)
    names = sheets(path)
    member = names[sheet] if sheet else next(iter(names.values()))
    rows: List[List[Optional[str]]] = []
    with zipfile.ZipFile(path) as zf:
        ss = _shared(zf)
        with zf.open(member) as f:
            for _, el in ET.iterparse(f):
                if el.tag != NS + "row":
                    continue
                cells: Dict[int, Optional[str]] = {}
                for c in el.iter(NS + "c"):
                    t = c.get("t")
                    v = c.find(NS + "v")
                    if t == "s" and v is not None:
                        val = ss[int(v.text)]
                    elif t == "inlineStr":
                        val = "".join(x.text or "" for x in c.iter(NS + "t"))
                    else:
                        val = v.text if v is not None else None
                    cells[_col(c.get("r"))] = val
                width = max(cells) + 1 if cells else 0
                rows.append([cells.get(i) for i in range(width)])
                el.clear()
                if max_rows and len(rows) >= max_rows:
                    break
    return rows
