"""Minimal readers for the CDFI spreadsheets (.xlsx and .xlsb), standard library only.

The project environment has neither openpyxl nor pyxlsb, so these read cell values
straight from the workbook XML / BIFF12 records. They return each sheet as a list of
rows (lists of str / float / None) and are only meant for the CDFI files used here.
"""
from __future__ import annotations

import re
import struct
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _col(ref: str) -> int:
    n = 0
    for ch in re.match(r"[A-Z]+", ref).group(0):
        n = n * 26 + ord(ch) - 64
    return n - 1


def _sheet_targets(z: zipfile.ZipFile, workbook: str, rels: str) -> Dict[str, str]:
    relmap = {}
    for rel in ET.fromstring(z.read(rels)):
        relmap[rel.get("Id")] = rel.get("Target")
    out = {}
    if workbook.endswith(".xml"):
        ns_r = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
        for sh in ET.fromstring(z.read(workbook)).iter(_NS + "sheet"):
            out[sh.get("name")] = relmap[sh.get(ns_r)]
    else:
        for t, d in _records(z.read(workbook)):
            if t == 156:                                    # BrtBundleSh
                rid, o = _xlwstr(d, 8)
                name, _ = _xlwstr(d, o)
                out[name] = relmap[rid]
    return {k: "xl/" + v.lstrip("/").replace("xl/", "", 1) for k, v in out.items()}


def read_xlsx(path: Path, sheet: str) -> List[List[Optional[str]]]:
    """Rows of one worksheet of an .xlsx file (values as strings)."""
    z = zipfile.ZipFile(path)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).iter(_NS + "si"):
            shared.append("".join(t.text or "" for t in si.iter(_NS + "t")))
    target = _sheet_targets(z, "xl/workbook.xml", "xl/_rels/workbook.xml.rels")[sheet]
    rows = []
    with z.open(target) as fh:
        for _, el in ET.iterparse(fh):
            if el.tag != _NS + "row":
                continue
            row = {}
            for c in el.iter(_NS + "c"):
                t, v = c.get("t"), c.find(_NS + "v")
                if t == "inlineStr":
                    val = "".join(x.text or "" for x in c.iter(_NS + "t"))
                elif v is None:
                    val = None
                elif t == "s":
                    val = shared[int(v.text)]
                else:
                    val = v.text
                row[_col(c.get("r"))] = val
            rows.append([row.get(i) for i in range(max(row) + 1)] if row else [])
            el.clear()
    return rows


def _records(b: bytes):
    i, n = 0, len(b)
    while i < n:
        t = b[i]
        i += 1
        if t & 0x80:
            t = (t & 0x7F) | (b[i] << 7)
            i += 1
        size = shift = 0
        for _ in range(4):
            c = b[i]
            i += 1
            size |= (c & 0x7F) << shift
            shift += 7
            if not c & 0x80:
                break
        yield t, b[i:i + size]
        i += size


def _xlwstr(d: bytes, o: int):
    n = struct.unpack_from("<I", d, o)[0]
    return d[o + 4:o + 4 + 2 * n].decode("utf-16-le"), o + 4 + 2 * n


def _rk(v: int) -> float:
    if v & 2:
        x = v >> 2
        if x & (1 << 29):
            x -= 1 << 30
        f = float(x)
    else:
        f = struct.unpack("<d", struct.pack("<Q", (v & 0xFFFFFFFC) << 32))[0]
    return f / 100 if v & 1 else f


def read_xlsb(path: Path, sheet: str) -> List[List[object]]:
    """Rows of one worksheet of an .xlsb (BIFF12) file."""
    z = zipfile.ZipFile(path)
    shared = [_xlwstr(d, 1)[0] for t, d in _records(z.read("xl/sharedStrings.bin")) if t == 19]
    target = _sheet_targets(z, "xl/workbook.bin", "xl/_rels/workbook.bin.rels")[sheet]
    rows: Dict[int, Dict[int, object]] = {}
    r = -1
    for t, d in _records(z.read(target)):
        if t == 0:                                          # BrtRowHdr
            r = struct.unpack_from("<I", d, 0)[0]
        elif 1 <= t <= 11:                                  # cell records
            c = struct.unpack_from("<I", d, 0)[0]
            if t == 2:
                v = _rk(struct.unpack_from("<I", d, 8)[0])
            elif t in (5, 9):
                v = struct.unpack_from("<d", d, 8)[0]
            elif t == 7:
                v = shared[struct.unpack_from("<I", d, 8)[0]]
            elif t in (6, 8):
                v = _xlwstr(d, 8)[0]
            elif t in (4, 10):
                v = bool(d[8])
            else:
                v = None
            rows.setdefault(r, {})[c] = v
    return [[rows[i].get(c) for c in range(max(rows[i]) + 1)] for i in sorted(rows)]
